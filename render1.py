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
from gaussian_renderer import render_rade as render
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio

def convert_array_to_pil(depth_map):
    # Input: depth_map -> HxW numpy array with depth values 
    # Output: colormapped_im -> HxW numpy array with colorcoded depth values
    import numpy as np
    import matplotlib as mpl
    mask = depth_map!=0
    disp_map = 1/depth_map
    vmax = np.percentile(disp_map[mask], 95)
    vmin = np.percentile(disp_map[mask], 5)
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap='magma')
    mask = np.repeat(np.expand_dims(mask,-1), 3, -1)
    colormapped_im = (mapper.to_rgba(disp_map)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im[~mask] = 255
    return colormapped_im

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    single_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_single")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(single_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering0 = render(view, gaussians, pipeline, background, require_depth=True, dual_alpha=True)
        rendering1 = render(view, gaussians, pipeline, background, require_depth=True, dual_alpha=False)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering0["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        imageio.imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + "_depth.png"), convert_array_to_pil(rendering0['expected_depth'][0].cpu().numpy()))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(rendering["normal"]*0.5+0.5, os.path.join(render_path, '{0:05d}'.format(idx) + "_normal.png"))
        torchvision.utils.save_image(rendering1["render"], os.path.join(single_path, '{0:05d}'.format(idx) + ".png"))
        imageio.imwrite(os.path.join(single_path, '{0:05d}'.format(idx) + "_depth.png"), convert_array_to_pil(rendering1['expected_depth'][0].cpu().numpy()))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, optimizer_type="default", rendering_mode="abs")
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    exit()
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)