#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import os, random, time
from random import randint
from lpipsPyTorch import lpips
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render_rade as render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import fov2focal
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from render import render_set as render_set
from render1 import render_set as render_set1
from metrics import evaluate

from utils.taming_utils import compute_gaussian_score, get_edges, get_count_array

import math
def depths_double_to_points(view, depthmap1, depthmap2):
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins_inv = torch.tensor(
        [[1/fx, 0.,-W/(2 * fx)],
        [0., 1/fy, -H/(2 * fy),],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W)+0.5, torch.arange(H)+0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0).reshape(3, -1).float().cuda()
    rays_d = intrins_inv @ points
    points1 = depthmap1.reshape(1,-1) * rays_d
    points2 = depthmap2.reshape(1,-1) * rays_d
    return points1.reshape(3,H,W), points2.reshape(3,H,W)

def point_double_to_normal(view, points1, points2):
    points = torch.stack([points1, points2],dim=0)
    output = torch.zeros_like(points)
    dx = points[...,2:, 1:-1] - points[...,:-2, 1:-1]
    dy = points[...,1:-1, 2:] - points[...,1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=1), dim=1)
    output[...,1:-1, 1:-1] = normal_map
    return output

def depth_double_to_normal(view, depth1, depth2):
    points1, points2 = depths_double_to_points(view, depth1, depth2)
    return point_double_to_normal(view, points1, points2)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, websockets, score_coefficients, args):
    first_iter = 0
    densify_iter_num = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    pairs0, predepth, grid = torch.load(dataset.source_path+'/pairs_uni.pth', weights_only=False)
    grid = grid.cuda()
    name = {view.image_name:i for i,view in enumerate(trainCameras)}
    pairs = [[name[p[0]], name[p[1]]] for p in pairs0 if p[0] in name and p[1] in name]
    # pairs = pairs+np.random.randint(len(trainCameras), size=(len(pairs), 2)).tolist()
    # pairs = np.random.randint(len(trainCameras), size=(50000, 2)).tolist()

    viewpoint_stack = None
    
    # record time
    time_taken = {
            "forward": [],
            "backward": [],
            "step": []
    }
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    all_edges = []
    for view in scene.getTrainCameras():
        edges_loss = get_edges(view.original_image).squeeze().cuda()
        edges_loss_norm = (edges_loss - torch.min(edges_loss)) / (torch.max(edges_loss) - torch.min(edges_loss))
        all_edges.append(edges_loss_norm.cpu())

    counts_array = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    bg = torch.rand((3), device="cuda") if opt.random_background else background
    start = time.time()
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        if counts_array == None:
            counts_array = get_count_array(len(scene.gaussians.get_xyz), max(args.budget, len(scene.gaussians.get_xyz)*2), opt, mode=args.mode)
            print(counts_array)

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = pairs.copy()
            p = 1/np.unique(pairs, return_counts=True)[1]
            p /= p.sum()
            viewpoint_stack = np.vstack((viewpoint_stack, np.random.choice(np.arange(len(trainCameras)), size=int(len(pairs))*2, p=p).reshape(-1,2))).tolist()
        pair = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        view0, view1 = trainCameras[pair[0]], trainCameras[pair[1]]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        if args.benchmark_dir:
            start_time.record()

        render_pkg0 = render(view0, gaussians, pipe, bg, require_depth=False, dual_alpha=((iteration>15000) if args.dual else False))
        render_pkg1 = render(view1, gaussians, pipe, bg, require_depth=False, dual_alpha=((iteration>15000) if args.dual else False))
        image0, viewspace_point_tensor0, visibility_filter0, radii0 = render_pkg0["render"], render_pkg0["viewspace_points"], render_pkg0["visibility_filter"], render_pkg0["radii"]
        image1, viewspace_point_tensor1, visibility_filter1, radii1 = render_pkg1["render"], render_pkg1["viewspace_points"], render_pkg1["visibility_filter"], render_pkg1["radii"]

        # Loss
        gt_image0 = view0.original_image.cuda()
        gt_image1 = view1.original_image.cuda()
        Ll1 = l1_loss(image0, gt_image0)+l1_loss(image1, gt_image1)
        ssim_value = fast_ssim(image0.unsqueeze(0), gt_image0.unsqueeze(0))+fast_ssim(image0.unsqueeze(0), gt_image0.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (2.0 - ssim_value)
        
        if iteration>15000 and args.dual:
            if args.single_loss:
                render_pkg_single0 = render(view0, gaussians, pipe, bg, require_depth=True, dual_alpha=False)
                render_pkg_single1 = render(view1, gaussians, pipe, bg, require_depth=True, dual_alpha=False)
                loss = loss + l1_loss(render_pkg_single0["render"], gt_image0) + l1_loss(render_pkg_single1["render"], gt_image1)
                
            if opt.lambda_depth_normal > 0:
                rendered_expected_depth0: torch.Tensor = render_pkg_single0["expected_depth"]
                rendered_median_depth0: torch.Tensor = render_pkg_single0["median_depth"]
                rendered_normal0: torch.Tensor = render_pkg_single0["normal"]
                depth_middepth_normal0 = depth_double_to_normal(view0, rendered_expected_depth0, rendered_median_depth0)
                rendered_expected_depth1: torch.Tensor = render_pkg_single1["expected_depth"]
                rendered_median_depth1: torch.Tensor = render_pkg_single1["median_depth"]
                rendered_normal1: torch.Tensor = render_pkg_single1["normal"]
                depth_middepth_normal1 = depth_double_to_normal(view1, rendered_expected_depth1, rendered_median_depth1)
                depth_ratio = 0.6
                normal_error_map = (1 - (rendered_normal0.unsqueeze(0) * depth_middepth_normal0).sum(dim=1)) + (1 - (rendered_normal1.unsqueeze(0) * depth_middepth_normal1).sum(dim=1))
                depth_normal_loss = (1-depth_ratio) * normal_error_map[0].mean() + depth_ratio * normal_error_map[1].mean()
                loss = loss + depth_normal_loss*opt.lambda_depth_normal

            if opt.lambda_depth_pair > 0 and pair in pairs:
                depth0, depth1 = render_pkg_single0["expected_depth"][0], render_pkg_single1["expected_depth"][0]
                imgshape0, imgshape1 = depth0.shape, depth1.shape
                fx0, fy0, fx1, fy1 = fov2focal(view0.FoVx, imgshape0[1]), fov2focal(view0.FoVy, imgshape0[0]), fov2focal(view1.FoVx, imgshape1[1]), fov2focal(view1.FoVy, imgshape1[0])
                gy0, gx0 = torch.meshgrid(torch.linspace(-(imgshape0[0]-1)/2,(imgshape0[0]-1)/2,imgshape0[0]), torch.linspace(-(imgshape0[1]-1)/2,(imgshape0[1]-1)/2,imgshape0[1]))
                gy1, gx1 = torch.meshgrid(torch.linspace(-(imgshape1[0]-1)/2,(imgshape1[0]-1)/2,imgshape1[0]), torch.linspace(-(imgshape1[1]-1)/2,(imgshape1[1]-1)/2,imgshape1[1]))
                gx0, gy0, gx1, gy1 = gx0/fx0, gy0/fy0, gx1/fx1, gy1/fy1
                gx0, gy0, gx1, gy1 = gx0.cuda(), gy0.cuda(), gx1.cuda(), gy1.cuda()
                X0 = torch.stack((gx0*depth0,gy0*depth0,depth0))
                X1 = torch.stack((gx1*depth1,gy1*depth1,depth1))
                pose0, pose1 = view0.world_view_transform, view1.world_view_transform
                Xw0 = pose0[:3,:3]@X0.view(3,-1)+pose0.T.inverse()[:3,3:]
                Xw1 = pose1[:3,:3]@X1.view(3,-1)+pose1.T.inverse()[:3,3:]
                X1_0 = pose0.T[:3,:3]@Xw1+pose0.T[:3,3:]
                X0_1 = pose1.T[:3,:3]@Xw0+pose1.T[:3,3:]
                i0,j0 = (X1_0[:2]/X1_0[2:])*torch.Tensor([[fx0],[fy0]]).cuda()
                i1,j1 = (X0_1[:2]/X0_1[2:])*torch.Tensor([[fx1],[fy1]]).cuda()
                i0,j0,i1,j1 = 2*i0/imgshape0[1], 2*j0/imgshape0[0], 2*i1/imgshape1[1], 2*j1/imgshape1[0]
                depth0_1 = torch.nn.functional.grid_sample(depth0[None,None], torch.stack((i0,j0)).T[None,None])[0,0,0].view(imgshape0)
                depth1_0 = torch.nn.functional.grid_sample(depth1[None,None], torch.stack((i1,j1)).T[None,None])[0,0,0].view(imgshape1)
                depth_pairs_loss = (depth1_0-depth0).clamp(0).mean()+(depth0_1-depth1).clamp(0).mean()
                loss = loss + depth_pairs_loss*opt.lambda_depth_pair
            
            if opt.lambda_depth > 0 and pair in pairs:            
                dgt = predepth[np.where((pairs0==[view0.image_name, view1.image_name]).all(axis=1))[0].item()].cuda()
                d0 = torch.nn.functional.grid_sample(depth0[None,None], grid, align_corners=True)[0,0]
                d1 = torch.nn.functional.grid_sample(depth1[None,None], grid, align_corners=True)[0,0]
                depth_gt_loss = ((dgt[...,0]-d0).abs()*dgt[...,1]).sum()/d0.numel()+((dgt[...,2]-d1).abs()*dgt[...,3]).sum()/d1.numel()
                loss = loss + depth_gt_loss*opt.lambda_depth

        if args.half:
            loss = loss*0.5

        if args.benchmark_dir:
            end_time.record()
            torch.cuda.synchronize()
            time_taken["forward"] += [start_time.elapsed_time(end_time)]

        if args.benchmark_dir:
            start_time.record()
        loss.backward()
        if args.benchmark_dir:
            end_time.record()
            torch.cuda.synchronize()
            time_taken["backward"] += [start_time.elapsed_time(end_time)]

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter0] = torch.max(gaussians.max_radii2D[visibility_filter0], radii0[visibility_filter0])
                gaussians.add_densification_stats(viewspace_point_tensor0, visibility_filter0)
                gaussians.max_radii2D[visibility_filter1] = torch.max(gaussians.max_radii2D[visibility_filter1], radii1[visibility_filter1])
                gaussians.add_densification_stats(viewspace_point_tensor1, visibility_filter1)
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    my_viewpoint_stack = scene.getTrainCameras().copy()
                    edges_stack = all_edges.copy()

                    num_cams = args.cams
                    if args.cams == -1:
                        num_cams = len(my_viewpoint_stack)
                    edge_losses = []
                    camlist = []
                    for _ in range(num_cams):
                        loc = random.randint(0, len(my_viewpoint_stack) - 1)
                        camlist.append(my_viewpoint_stack.pop(loc))
                        edge_losses.append(edges_stack.pop(loc))

                    gaussian_importance = compute_gaussian_score(scene, camlist, edge_losses, gaussians, pipe, bg, score_coefficients, opt)                    
                    gaussians.densify_with_score(scores = gaussian_importance, 
                                                max_screen_size = size_threshold, 
                                                min_opacity = 0.005, 
                                                extent = scene.cameras_extent, 
                                                budget=counts_array[densify_iter_num+1], 
                                                radii=torch.max(radii0, radii1),
                                                nogof=args.nogof,
                                                iter_num=densify_iter_num)
                    densify_iter_num += 1
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration == args.ho_iteration:
                print("Release opacity limit")
                gaussians.modify_functions()

            # Optimizer step
            if args.benchmark_dir:
                start_time.record()
            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    if args.sh_lower:
                        if iteration % 16 == 0:
                            gaussians.shoptimizer.step()
                            gaussians.shoptimizer.zero_grad(set_to_none = True)
                    else:
                        gaussians.shoptimizer.step()
                        gaussians.shoptimizer.zero_grad(set_to_none = True)
                elif opt.optimizer_type == "sparse_adam":
                    visible = torch.max(radii0, radii1) > 0
                    gaussians.optimizer.step(visible, torch.max(radii0, radii1).shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
            if args.benchmark_dir:
                end_time.record()
                torch.cuda.synchronize()
                time_taken["step"] += [start_time.elapsed_time(end_time)]

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                my_viewpoint_stack = scene.getTrainCameras().copy()

                num_cams = args.cams
                if args.cams == -1:
                    num_cams = len(my_viewpoint_stack)
                camlist = []
                for _ in range(num_cams):
                    camlist.append(my_viewpoint_stack.pop(random.randint(0, len(my_viewpoint_stack) - 1)))

                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        if args.benchmark_dir:
            os.makedirs(args.benchmark_dir, exist_ok = True)
            np.save(os.path.join(args.benchmark_dir, "forward.npy"), np.array(time_taken["forward"]))
            np.save(os.path.join(args.benchmark_dir, "backward.npy"), np.array(time_taken["backward"]))
            np.save(os.path.join(args.benchmark_dir, "step.npy"), np.array(time_taken["step"]))

    end = time.time()
    scene.save(iteration)
    print(f"Time taken by {os.getenv('OAR_JOB_ID')}: {end - start}s")
    
    with torch.no_grad():
        if args.dual:
            render_set1(dataset.model_path, "test", iteration, scene.getTestCameras(), gaussians, pipe, background)
        else:
            render_set(dataset.model_path, "test", iteration, scene.getTestCameras(), gaussians, pipe, background)
        evaluate([dataset.model_path])

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--cams", type=int, default=10)
    parser.add_argument("--budget", type=float, default=20)
    parser.add_argument("--mode", type=str, default="multiplier", choices=["multiplier", "final_count"])
    parser.add_argument("--websockets", action='store_true', default=False)
    parser.add_argument("--ho_iteration", type=int, default=15000)
    parser.add_argument("--sh_lower", action='store_true', default=False)
    parser.add_argument("--benchmark_dir", type=str, default=None)
    parser.add_argument("--dual", action='store_true', default=False)
    parser.add_argument("--single_loss", action='store_true', default=False)
    parser.add_argument("--half", action='store_true', default=False)
    parser.add_argument("--nogof", action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    score_coefficients = {'view_importance': 50, 'edge_importance': 50, 'mse_importance': 50, 'grad_importance': 25, 'dist_importance': 50, 'opac_importance': 100, 'dept_importance': 5, 'loss_importance': 10, 'radii_importance': 10, 'scale_importance': 25, 'count_importance': 0.1, 'blend_importance': 50}
    # exit()
    training(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from, 
        args.websockets, 
        score_coefficients, 
        args
    )

    # All done
    print("\nTraining complete.")
