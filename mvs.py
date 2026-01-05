import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import cv2
import pycolmap
import networkx as nx
import g2o
import trimesh
import open3d as o3d
import warnings
from argparse import Namespace
from scipy.spatial import KDTree
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)

def get_magsac_config(ransacReprojThreshold=2.0, confidence=0.99, maxIters=1000):
    usac_params = cv2.UsacParams()

    usac_params.randomGeneratorState = np.random.randint(0,1000000)
    usac_params.confidence = confidence
    usac_params.maxIterations = maxIters
    usac_params.loMethod = cv2.LOCAL_OPTIM_SIGMA
    usac_params.score = cv2.SCORE_METHOD_MAGSAC
    usac_params.threshold = ransacReprojThreshold
    usac_params.isParallel = True # False is deafult
    usac_params.loIterations = 10
    usac_params.loSampleSize = 50
    usac_params.neighborsSearch = cv2.NEIGH_GRID
    usac_params.sampler = cv2.SAMPLING_UNIFORM
    
    return usac_params

def signed_expm1(x):
    sign = torch.sign(x)
    return sign * torch.expm1(torch.abs(x))
def cosine_schedule(t, lr_start, lr_end):
    # assert 0 <= t <= 1
    t = np.clip(t, 0, 1)
    return lr_end + (lr_start - lr_end) * (1+np.cos(t * np.pi))/2
def adjust_learning_rate_by_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr  

class TestDataset(Dataset):
    def __init__(self, dataset_folder):
        self.root = dataset_folder
        self.image_paths = sorted([x for x in os.listdir(dataset_folder) if '.jpg' in x or '.png' in x or '.JPG' in x or 'PNG' in x])
        self.trans0 = None
        self.trans = None

    def need_ori_img(self, need:bool=True):
        if need:
            self.trans0 = transforms.Compose([
                # transforms.Resize(size=1600,max_size=2560),
                transforms.ToTensor()
                ])

    def set_trans(self, trans):
        self.trans = transforms.Compose(trans)

    @staticmethod
    def open_image(path):
        return Image.open(path).convert("RGB")
    
    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.image_paths[index])
        pil_img = TestDataset.open_image(image_path)
        return self.trans0(pil_img) if self.trans0 else 0, self.trans(pil_img) if self.trans else 0, index
    
    def __len__(self):
        return len(self.image_paths)

def cosplace(images):
    if not hasattr(cosplace, 'model'):
        cosplace.backbone = 'ResNet152'
        cosplace.dim = 2048
        cosplace.model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone=cosplace.backbone, fc_output_dim=cosplace.dim)
        cosplace.model.eval().cuda()
    if isinstance(images, torch.Tensor) and images.device.type=='cuda':
        return cosplace.model(images)

def boq(images):
    if not hasattr(boq, 'model'):
        boq.model = torch.hub.load("amaralibey/bag-of-queries", "get_trained_boq", backbone_name="dinov2", output_dim=12288, verbose=False).eval().cuda()
        boq.dim = 12288
    if isinstance(images, torch.Tensor) and images.device.type=='cuda':
        return boq.model(images)[0]

def salad(images):
    if not hasattr(salad, 'model'):
        salad.model = torch.hub.load("serizba/salad", "dinov2_salad").eval().cuda()
        salad.dim = 8448
    if isinstance(images, torch.Tensor) and images.device.type=='cuda':
        return salad.model(images)

def method_init(method):
    if method == 'cosplace':
        descriptor = cosplace
        trans = [
            transforms.Resize(960, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        descriptor(None)
        return descriptor, descriptor.dim, trans
    if method == 'boq':
        descriptor = boq
        trans = [
            transforms.ToTensor(),
            transforms.Resize((322, 322), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        descriptor(None)
        return descriptor, descriptor.dim, trans
    if method == 'salad':
        descriptor = salad
        trans = [
            transforms.Resize((322, 322), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        descriptor(None)
        return descriptor, descriptor.dim, trans
    else:
        return None, None, None

class Dust3r:
    def __init__(self, dust3r_path):
        sys.path.append(dust3r_path)
        from dust3r.model import AsymmetricCroCo3DStereo
        model_name = os.path.join(dust3r_path, "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
        # you can put the path to a local checkpoint in model_name if needed
        self.model = AsymmetricCroCo3DStereo.from_pretrained(model_name).cuda()
    
    def encode(self, x):
        out, pos, _ = self.model._encode_image(x, x.shape[-2:])
        return out, pos
    
    def decode(self, feat1, pos1, feat2, pos2, shape1, shape2):
        dec1, dec2 = self.model._decoder(feat1, pos1, feat2, pos2)

        with torch.amp.autocast(enabled=False, device_type='cuda'):
            res1 = self.model._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self.model._downstream_head(2, [tok.float() for tok in dec2], shape2)
        return res1, res2
    
if __name__ == '__main__':
    path = sys.argv[1]
    dust3r_path = 'your_path_to_dust3r'
    assert os.path.exists(dust3r_path)
    batch_size = 8
    trans_match = True

    desc_thmax = 0.8
    desc_thres = 0.3
    desc_thmin = 0.1

    recon = pycolmap.Reconstruction()
    recon.read(path+'/../sparse/0')

    dataset = TestDataset(path)
    recon_images = {}
    j = 1
    for i in range(max(list(recon.images.keys()))):
        if i+1 in recon.images:
            recon_images[j] = recon.images[i+1]
            j += 1
    dataset.image_paths = [recon_images[i+1].name for i in range(max(list(recon_images.keys())))]
    N = len(dataset)
    
    dataset.need_ori_img()
    imgheight, imgwidth = dataset[0][0].shape[1:]
    descriptor, dim, trans = method_init('boq')
    dataset.set_trans(trans)
    descs = torch.zeros((len(dataset), 4, dim), dtype=torch.float32).cuda() # origin, left rotate, right rotate, rotate 180
    dataloader = DataLoader(dataset=dataset, num_workers=12, batch_size=batch_size, pin_memory=True)
    for images, images_resized, indices in tqdm(dataloader):
        desc = descriptor(images_resized.cuda())
        descs[indices, 0] = desc
        if trans_match:
            desc = descriptor(images_resized.permute(0,1,3,2).flip(dims=(2,)).cuda())
            descs[indices, 1] = desc
            desc = descriptor(images_resized.permute(0,1,3,2).flip(dims=(3,)).cuda())
            descs[indices, 2] = desc
            desc = descriptor(images_resized.flip(dims=(2,)).flip(dims=(3,)).cuda())
            descs[indices, 3] = desc
    del descriptor.model, descriptor, dim, trans

    resp = torch.stack((descs[:,0]@descs[:,0].T, descs[:,0]@descs[:,1].T, descs[:,0]@descs[:,2].T, descs[:,0]@descs[:,3].T))
    del descs
    resp = resp.triu()+resp.triu().permute(0,2,1)
    resp[0] += 0.1
    resp, resp_idx = resp.max(dim=0)
    resp[resp_idx==0] -= 0.1
    resp[torch.arange(N), torch.arange(N)] /= 2
    resp_idx = resp_idx.cpu()
    x = resp.clone()
    y = torch.zeros_like(resp)
    thmax = torch.Tensor([desc_thmax]).cuda()
    resp_k = resp.topk(7, dim=0)[0][1:]
    x[x>desc_thmax] = 0
    i = 0
    while 1:
        idx = x.argmax(dim=0)
        y[torch.arange(N), idx] += 1
        x[torch.arange(N), idx] = 0
        x[resp[idx]>desc_thmax] = 0
        x = x.minimum(x.T)
        pairs = torch.stack(torch.where((y+y.T).triu())).T.cpu()
        G = nx.Graph()
        G.add_edges_from(pairs.numpy())
        i += 1
        if nx.is_connected(G): break
    # while (x>desc_thmin).any():
    for i in range(i,max(30,i+15)):
        # if (i%3==0) and resp_k.any(): thmax = thmax.minimum(resp_k[1]-0.001); resp_k = resp_k[2:]
        if resp_k.any(): thmax = thmax.minimum(resp_k[0]-0.001); resp_k = resp_k[1:]
        idx = x.argmax(dim=0)
        y[torch.arange(N), idx] += (x[torch.arange(N), idx]>desc_thmin)*1.0
        x[torch.arange(N), idx] = 0
        x[resp[idx]>thmax[idx,None]] = 0
        x = x.minimum(x.T)
    for i in range(i,60):
        if not (x>desc_thres).any(): break
        idx = x.argmax(dim=0)
        y[torch.arange(N), idx] += (x[torch.arange(N), idx]>desc_thres)*1.0
        x[torch.arange(N), idx] = 0
        x[resp[idx]>thmax[idx,None]] = 0
        x = x.minimum(x.T)
    pairs = torch.stack(torch.where((y+y.T).triu())).T.cpu()

    fx, fy, cx, cy = recon.cameras[1].params[:4]
    fx *= imgwidth/2/cx
    fy *= imgheight/2/cy
    cx, cy = imgwidth/2, imgheight/2
    dist_par = recon.cameras[1].params[4:]
    K = np.diag([fx, fy, 1])
    K[0,2] = cx
    K[1,2] = cy
    # vaild_frame = [i for i in range(N) if i+1 in recon.images]
    # invaild_frame = np.setdiff1d(np.arange(N),vaild_frame)
    # for i in invaild_frame: pairs[np.where(pairs==i)[0]] = -1
    # pairs = pairs[pairs[:,0]>=0]
    w2c = torch.from_numpy(np.concatenate((np.stack([recon_images[i+1].cam_from_world.matrix for i in range(N)]), np.array([0,0,0,1.]).reshape((1,1,4)).repeat(N,axis=0)), axis=1)).float()
    dpose = w2c[pairs[:,1]]@w2c.inverse()[pairs[:,0]]
    dr = (torch.arccos(dpose[:,[0,1,2],[0,1,2]].sum(dim=1)*0.5-0.5)/np.pi*180).abs()
    dr = dr.minimum(180-dr)
    dz = dpose[:,2,3].abs()/dpose[:,:3,3].norm(dim=1)
    dr_th = torch.ones_like(dr)*16
    for i in range(N):
        idx = np.where(pairs==i)[0]
        dr_th[idx] = np.minimum(dr[idx][(dr[idx]<60)&(dz[idx]<0.9)].sort()[0][-2:][0].item()-0.001, dr_th[idx])
    pairs = pairs[(dr>dr_th)&(dr<60)&(dz<0.9)]
    
    rate = min(512/max(imgheight, imgwidth), 1)
    imgheight_new, imgwidth_new = int((imgheight*rate+0.5)//16*16), int((imgwidth*rate+0.5)//16*16)
    cx, cy = imgwidth_new/2, imgheight_new/2
    K_new = np.diag([fx*rate, fy*rate, 1])
    K_new[0,2] = cx
    K_new[1,2] = cy
    
    images_undist = torch.zeros((N,3,imgheight_new,imgwidth_new))
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist_par, None, K_new, (imgwidth_new,imgheight_new), cv2.CV_32FC1)
    dataset.trans = None
    dataloader = DataLoader(dataset=dataset, num_workers=12, batch_size=batch_size, pin_memory=True)
    for images, images_resized, indices in tqdm(dataloader):
        for image, i in zip(images, indices):
            images_undist[i] = torch.from_numpy(cv2.remap(image.permute(1,2,0).numpy(), mapx, mapy, cv2.INTER_CUBIC).clip(0,1).transpose((2,0,1))*2-1).float()

    dust3r = Dust3r(dust3r_path)
    dust_batch_size = 16
    imgshape = torch.Tensor([[imgheight_new, imgwidth_new]]).long()
    dust_feats = torch.zeros((N, images_undist[0,0].numel()//256, 1024))
    dust_pos = torch.zeros((N, images_undist[0,0].numel()//256, 2), dtype=torch.int64)
    for i in tqdm(range(0, N, dust_batch_size)):
        out, pos = dust3r.encode(images_undist[i:i+dust_batch_size].cuda())
        dust_feats[i:i+dust_batch_size] = out
        dust_pos[i:i+dust_batch_size] = pos
    add_frame_left, add_index_left = torch.unique(pairs[resp_idx[pairs[:,0],pairs[:,1]]==1][:,1], return_inverse=True)
    add_dust_feats_left = torch.zeros((len(add_frame_left), images_undist[0,0].numel()//256, 1024))
    add_dust_pos_left = torch.zeros((len(add_frame_left), images_undist[0,0].numel()//256, 2), dtype=torch.int64)
    for i in tqdm(range(0, len(add_frame_left), dust_batch_size)):
        out, pos = dust3r.encode(images_undist[add_frame_left[i:i+dust_batch_size]].permute(0,1,3,2).flip(dims=(2,)).cuda())
        add_dust_feats_left[i:i+dust_batch_size] = out
        add_dust_pos_left[i:i+dust_batch_size] = pos
    add_frame_right, add_index_right = torch.unique(pairs[resp_idx[pairs[:,0],pairs[:,1]]==2][:,1], return_inverse=True)
    add_dust_feats_right = torch.zeros((len(add_frame_right), images_undist[0,0].numel()//256, 1024))
    add_dust_pos_right = torch.zeros((len(add_frame_right), images_undist[0,0].numel()//256, 2), dtype=torch.int64)
    for i in tqdm(range(0, len(add_frame_right), dust_batch_size)):
        out, pos = dust3r.encode(images_undist[add_frame_right[i:i+dust_batch_size]].permute(0,1,3,2).flip(dims=(3,)).cuda())
        add_dust_feats_right[i:i+dust_batch_size] = out
        add_dust_pos_right[i:i+dust_batch_size] = pos
    add_frame_180, add_index_180 = torch.unique(pairs[resp_idx[pairs[:,0],pairs[:,1]]==3][:,1], return_inverse=True)
    add_dust_feats_180 = torch.zeros((len(add_frame_180), images_undist[0,0].numel()//256, 1024))
    add_dust_pos_180 = torch.zeros((len(add_frame_180), images_undist[0,0].numel()//256, 2), dtype=torch.int64)
    for i in tqdm(range(0, len(add_frame_180), dust_batch_size)):
        out, pos = dust3r.encode(images_undist[add_frame_180[i:i+dust_batch_size]].flip(dims=(2,)).flip(dims=(3,)).cuda())
        add_dust_feats_180[i:i+dust_batch_size] = out
        add_dust_pos_180[i:i+dust_batch_size] = pos
    trans_len = torch.hstack([torch.zeros(1,dtype=torch.long)]+[(resp_idx[pairs[:,0],pairs[:,1]]==i).sum() for i in range(4)]+[torch.zeros(1,dtype=torch.long)]).cumsum(dim=0)
    pairs = torch.vstack((pairs[resp_idx[pairs[:,0],pairs[:,1]]==0], pairs[resp_idx[pairs[:,0],pairs[:,1]]==1], pairs[resp_idx[pairs[:,0],pairs[:,1]]==2], pairs[resp_idx[pairs[:,0],pairs[:,1]]==3]))
    # del images_undist
    
    dust_batch_size = 8
    gy, gx = torch.meshgrid(torch.linspace(-(imgshape[0,0]-1)/2,(imgshape[0,0]-1)/2,imgshape[0,0]), torch.linspace(-(imgshape[0,1]-1)/2,(imgshape[0,1]-1)/2,imgshape[0,1]))
    g = torch.stack((gx,gy)).permute(1,2,0)[None].cuda()

    query_res = torch.zeros(len(pairs), 4)
    data = torch.empty(len(pairs),*imgshape[0],4)
    Tp = torch.zeros(len(pairs), 4, 4)
    x = torch.zeros(len(pairs), 3, 7)
    y = torch.zeros(len(pairs), 3, 7)
    w = torch.zeros(len(pairs), 1, 7)
    n = torch.zeros(len(pairs), 2, 2)
    for trans in range(4):
        start, end = trans_len[[trans, trans+1]]
        start, end = start.item(), end.item()
        for i in tqdm(range(start, end, dust_batch_size)):
            index = pairs[i:min(i+dust_batch_size, end)]
            feat1, pos1 = dust_feats[index[:,0]].cuda(), dust_pos[index[:,0]].cuda()
            shape1 = imgshape.repeat(len(feat1), 1)
            if trans:
                if trans==1:
                    add_index, add_dust_feats, add_dust_pos = add_index_left, add_dust_feats_left, add_dust_pos_left
                    shape2 = imgshape[:,[1,0]].repeat(len(feat1), 1)
                elif trans==2:
                    add_index, add_dust_feats, add_dust_pos = add_index_right, add_dust_feats_right, add_dust_pos_right
                    shape2 = imgshape[:,[1,0]].repeat(len(feat1), 1)
                else:
                    add_index, add_dust_feats, add_dust_pos = add_index_180, add_dust_feats_180, add_dust_pos_180
                    shape2 = imgshape.repeat(len(feat1), 1)
                idx = add_index[torch.arange(i,min(i+dust_batch_size, end))-trans_len[trans]]
                feat2 = add_dust_feats[idx].cuda()
                pos2 = add_dust_pos[idx].cuda()
            else:
                feat2, pos2 = dust_feats[index[:,1]].cuda(), dust_pos[index[:,1]].cuda()
                shape2 = imgshape.repeat(len(feat1), 1)
            res12 = dust3r.decode(feat1, pos1, feat2, pos2, shape1, shape2)
            res21 = dust3r.decode(feat2, pos2, feat1, pos1, shape2, shape1)
            if trans==1:
                res12[1]['pts3d'] = res12[1]['pts3d'].permute(0,2,1,3).flip(dims=(2,))
                res12[1]['conf'] = res12[1]['conf'].permute(0,2,1).flip(dims=(2,))
                res21[0]['pts3d'] = res21[0]['pts3d'].permute(0,2,1,3).flip(dims=(2,))[...,[1,0,2]]
                res21[0]['conf'] = res21[0]['conf'].permute(0,2,1).flip(dims=(2,))
                res21[1]['pts3d'] = res21[1]['pts3d'][...,[1,0,2]]
                res21[0]['pts3d'][...,0] *= -1
                res21[1]['pts3d'][...,0] *= -1
            elif trans==2:
                res12[1]['pts3d'] = res12[1]['pts3d'].permute(0,2,1,3).flip(dims=(1,))
                res12[1]['conf'] = res12[1]['conf'].permute(0,2,1).flip(dims=(1,))
                res21[0]['pts3d'] = res21[0]['pts3d'].permute(0,2,1,3).flip(dims=(1,))[...,[1,0,2]]
                res21[0]['conf'] = res21[0]['conf'].permute(0,2,1).flip(dims=(1,))
                res21[1]['pts3d'] = res21[1]['pts3d'][...,[1,0,2]]
                res21[0]['pts3d'][...,1] *= -1
                res21[1]['pts3d'][...,1] *= -1
            elif trans==3:
                res12[1]['pts3d'] = res12[1]['pts3d'].flip(dims=(1,)).flip(dims=(2,))
                res12[1]['conf'] = res12[1]['conf'].flip(dims=(1,)).flip(dims=(2,))
                res21[0]['pts3d'] = res21[0]['pts3d'].flip(dims=(1,)).flip(dims=(2,))
                res21[0]['conf'] = res21[0]['conf'].flip(dims=(1,)).flip(dims=(2,))
                res21[0]['pts3d'][...,:2] *= -1
                res21[1]['pts3d'][...,:2] *= -1
            res12[0]['conf'] -= 1-1e-5
            res21[0]['conf'] -= 1-1e-5
            res12[1]['conf'] -= 1-1e-5
            res21[1]['conf'] -= 1-1e-5
            
            # data[i:min(i+dust_batch_size, end)] = torch.concat((res12[0]['pts3d'], res12[0]['conf'][...,None], res21[0]['pts3d'], res21[0]['conf'][...,None], res12[1]['pts3d'], res12[1]['conf'][...,None], res21[1]['pts3d'], res21[1]['conf'][...,None]), dim=-1)
            data[i:min(i+dust_batch_size, end)] = torch.stack((res12[0]['pts3d'][...,2], res12[0]['conf'], res21[0]['pts3d'][...,2], res21[0]['conf'])).permute(1,2,3,0)

            th12 = torch.median(res12[0]['pts3d'][...,-1].reshape(len(index),-1), dim=1)[0]
            th21 = torch.median(res21[0]['pts3d'][...,-1].reshape(len(index),-1), dim=1)[0]        
            ii = (torch.stack((gx,gy)).cuda().permute(1,2,0).norm()/(res12[0]['pts3d'][...,:2]/res12[0]['pts3d'][...,2:]).reshape(len(index),-1).norm(dim=1))[:,None,None,None]*(res12[1]['pts3d'][...,:2]/res12[1]['pts3d'][...,2:])
            ii[...,0] += cx
            ii[...,1] += cy
            ii = (ii+0.5).long()
            ii[...,0][(ii[...,0]<0)|(ii[...,0]>=imgshape[0,1])] = -1
            ii[...,1][(ii[...,1]<0)|(ii[...,1]>=imgshape[0,0])] = -1
            ii[...,1] *= imgshape[0,1]
            ii[(ii<0).any(dim=-1)] = -1
            ii = ii[...,0]+ii[...,1]
            ii[ii<0] = imgshape.prod()
            ii = ii.reshape(len(index),-1)
            dd = torch.hstack((res12[0]['pts3d'][...,-1].reshape(len(index),-1),torch.ones(len(index),1).cuda()*-1000))[torch.arange(len(index))[:,None].cuda(),ii].reshape(len(index),*imgshape[0])
            w12 = torch.hstack((res12[0]['conf'].reshape(len(index),-1),torch.ones(len(index),1).cuda()*-1000))[torch.arange(len(index))[:,None].cuda(),ii].reshape(len(index),*imgshape[0])
            m12 = ((res12[1]['pts3d'][...,-1]-dd).abs()<th12[:,None,None]*0.01)&(w12>0)
            ii = (torch.stack((gx,gy)).cuda().permute(1,2,0).norm()/(res21[0]['pts3d'][...,:2]/res21[0]['pts3d'][...,2:]).reshape(len(index),-1).norm(dim=1))[:,None,None,None]*(res21[1]['pts3d'][...,:2]/res21[1]['pts3d'][...,2:])
            ii[...,0] += cx
            ii[...,1] += cy
            ii = (ii+0.5).long()
            ii[...,0][(ii[...,0]<0)|(ii[...,0]>=imgshape[0,1])] = -1
            ii[...,1][(ii[...,1]<0)|(ii[...,1]>=imgshape[0,0])] = -1
            ii[...,1] *= imgshape[0,1]
            ii[(ii<0).any(dim=-1)] = -1
            ii = ii[...,0]+ii[...,1]
            ii[ii<0] = imgshape.prod()
            ii = ii.reshape(len(index),-1)
            dd = torch.hstack((res21[0]['pts3d'][...,-1].reshape(len(index),-1),torch.ones(len(index),1).cuda()*-1000))[torch.arange(len(index))[:,None].cuda(),ii].reshape(len(index),*imgshape[0])
            w21 = torch.hstack((res21[0]['conf'].reshape(len(index),-1),torch.ones(len(index),1).cuda()*-1000))[torch.arange(len(index))[:,None].cuda(),ii].reshape(len(index),*imgshape[0])
            m21 = ((res21[1]['pts3d'][...,-1]-dd).abs()<th21[:,None,None]*0.01)&(w21>0)
            query_res[i:min(i+dust_batch_size, end),0] = ((w12*res12[1]['conf']/(w12+res12[1]['conf'])*m12).sum(dim=[1,2])+(w21*res21[1]['conf']/(w21+res21[1]['conf'])*m21).sum(dim=[1,2]))/imgshape.prod()
            query_res[i:min(i+dust_batch_size, end),1] = (m12.sum(dim=[1,2])+m21.sum(dim=[1,2]))/imgshape.prod()

            a = (res12[0]['pts3d'][:,...,:2]/res12[0]['pts3d'][:,...,2:])
            wx = res12[0]['conf'][:,:,1:]*res12[0]['conf'][:,:,:-1]/(res12[0]['conf'][:,:,1:]+res12[0]['conf'][:,:,:-1])
            wy = res12[0]['conf'][:,1:]*res12[0]['conf'][:,:-1]/(res12[0]['conf'][:,1:]+res12[0]['conf'][:,:-1])
            n[i:min(i+dust_batch_size, end),0] = torch.stack((((a[:,:,1:,0]-a[:,:,:-1,0])*wx).sum()/wx.sum(), ((a[:,1:,:,1]-a[:,:-1,:,1])*wy).sum()/wy.sum()))
            a = (res21[0]['pts3d'][:,...,:2]/res21[0]['pts3d'][:,...,2:])
            wx = res21[0]['conf'][:,:,1:]*res21[0]['conf'][:,:,:-1]/(res21[0]['conf'][:,:,1:]+res21[0]['conf'][:,:,:-1])
            wy = res21[0]['conf'][:,1:]*res21[0]['conf'][:,:-1]/(res21[0]['conf'][:,1:]+res21[0]['conf'][:,:-1])
            n[i:min(i+dust_batch_size, end),1] = torch.stack((((a[:,:,1:,0]-a[:,:,:-1,0])*wx).sum()/wx.sum(), ((a[:,1:,:,1]-a[:,:-1,:,1])*wy).sum()/wy.sum()))
            X = torch.concat((
                torch.concat((res12[0]['pts3d'].permute((0,3,1,2))[:,:,None], res12[1]['pts3d'].permute((0,3,1,2))[:,:,None]),dim=2), 
                torch.concat((res21[1]['pts3d'].permute((0,3,1,2))[:,:,None], res21[0]['pts3d'].permute((0,3,1,2))[:,:,None]),dim=2), 
                torch.concat((((res12[0]['conf']*res21[1]['conf'])/(res12[0]['conf']+res21[1]['conf']))[:,None], ((res12[1]['conf']*res21[0]['conf'])/(res12[1]['conf']+res21[0]['conf']))[:,None]), dim=1)[:,None]), dim=1)
            X = X.reshape(len(X), 7, -1)
            X[:,:6] *= X[:,6:]
            u,d,_ = torch.linalg.svd(X, full_matrices=False)
            u *= u[:,-1:].sign()
            d *= u[:,-1]
            u[:,:-1] /= u[:,-1:]
            x[i:min(i+dust_batch_size, end)],y[i:min(i+dust_batch_size, end)],w[i:min(i+dust_batch_size, end)] = u[:,:3],u[:,3:6],d[:,None]

            xx, yy, ww = x[i:min(i+dust_batch_size, end)].clone(),y[i:min(i+dust_batch_size, end)].clone(),w[i:min(i+dust_batch_size, end)].clone()
            xm = (xx*ww).sum(dim=-1, keepdim=True)/ww.sum(dim=-1, keepdim=True)
            ym = (yy*ww).sum(dim=-1, keepdim=True)/ww.sum(dim=-1, keepdim=True)
            xx -= xm; yy -= ym
            s = (((yy*ww)**2).sum(dim=[1,2])/((xx*ww)**2).sum(dim=[1,2])).sqrt()
            u,d,vt = torch.linalg.svd((yy*ww)@(xx*ww).permute(0,2,1))
            u[:,:,-1] *= (u@vt).det()[:,None]
            R = u@vt
            T = ym-R@xm*s[:,None,None]
            xx += xm; yy += ym
            Tp[i:min(i+dust_batch_size, end)] = torch.concat((torch.concat((s[:,None,None]*R, T), dim=2), torch.Tensor([0,0,0,1.]).reshape(1,1,4).repeat(len(s),1,1)), dim=1)

            m1to2 = ((R.cuda()@res12[1]['pts3d'].permute((0,3,1,2)).reshape(len(index),3,-1)*s[:,None,None].cuda()+T.cuda())[:,2].reshape(len(index),*imgshape[0])-res21[0]['pts3d'][...,-1]).abs()<th21[:,None,None]*0.01
            m2to1 = ((R.cuda().permute(0,2,1)@(res21[1]['pts3d'].permute((0,3,1,2)).reshape(len(index),3,-1)-T.cuda())/s[:,None,None].cuda())[:,2].reshape(len(index),*imgshape[0])-res12[0]['pts3d'][...,-1]).abs()<th12[:,None,None]*0.01
            w1to2 = (res12[1]['conf']*res21[0]['conf']/(res12[1]['conf']+res21[0]['conf'])*m1to2).sum(dim=[1,2])
            w2to1 = (res21[1]['conf']*res12[0]['conf']/(res21[1]['conf']+res12[0]['conf'])*m2to1).sum(dim=[1,2])
            query_res[i:min(i+dust_batch_size, end),2] = (w1to2+w2to1)/imgshape.prod()
            query_res[i:min(i+dust_batch_size, end),3] = (m1to2.sum(dim=[1,2])+m2to1.sum(dim=[1,2]))/imgshape.prod()
    data.clamp_(min=1e-5)

    del dust3r
    torch.cuda.empty_cache()
    dw = torch.ones(N)*0.5
    for i in range(N):
        dw[i] = min(dw[i], query_res[np.where(pairs==i)[0]][:,[0,2]].min(axis=1)[0].sort()[0][-2:][0]*0.8)
    mask = (query_res[:,[0,2]]>dw[pairs].min(axis=1)[0][:,None]).all(dim=1)#&(query_res[:,[1,3]]<1.8).all(dim=1)
    pairs = pairs[mask]
    data = data[mask]
    Tp = Tp[mask]
    x = x[mask]
    y = y[mask]
    w = w[mask]
    n = n[mask]
    
    M = len(pairs)
    x = torch.concat((x,torch.ones(len(x),1,x.shape[-1])), dim=1)
    y = torch.concat((y,torch.ones(len(y),1,y.shape[-1])), dim=1)
    x, y, w = x.cuda(), y.cuda(), w.cuda()
    
    torch.set_grad_enabled(True)
    RT0 = torch.from_numpy(np.linalg.inv(np.float64(w2c.numpy())).astype(np.float32)).cuda()
    scale = torch.nn.Parameter(torch.zeros(2*M).cuda())
    scale1 = torch.nn.Parameter(torch.zeros(2, 2*M).cuda())
    focal_scale = torch.nn.Parameter(torch.from_numpy(K_new.diagonal()[:2]).float().log().cuda())
    norm_fs = torch.vstack((n[:,0], n[:,1])).cuda()

    optimizer = torch.optim.Adam([scale, focal_scale, scale1], lr=0.01)
    L = []
    for i in tqdm(range(20000)):
        optimizer.zero_grad()
        lr = cosine_schedule(i/20000, 1e-2, 1e-4)
        adjust_learning_rate_by_lr(optimizer, lr)
        RT = torch.vstack((RT0[pairs[:,0]], RT0[pairs[:,1]]))
        S = scale.exp()#/(scale.mean()).exp()
        S1 = scale1.exp()
        SRT = RT*torch.stack((S*S1[0],S*S1[1],S,torch.ones_like(S))).T[:,None]
        r = ((SRT[:M]@x)[:,:-1]-(SRT[M:2*M]@y)[:,:-1])*w
        loss = (r.norm(dim=[1,2])).log1p().mean()
        loss += ((S1.T*focal_scale.exp())*norm_fs-1).abs().mean()*1000
        L.append(loss.item())
        loss.backward()
        optimizer.step()
    
    kpt_uv = []
    for i in tqdm(range(N)):
        img = recon_images[i+1]
        uvs = np.round((np.stack([pt.xy for pt in img.points2D])/np.mean((recon.cameras[1].width/imgwidth, recon.cameras[1].height/imgheight))-np.array([imgwidth/2, imgheight/2]))*rate+np.array([imgwidth_new/2, imgheight_new/2])).astype(np.int32)
        ids = np.uint64([pt.point3D_id for pt in img.points2D])
        ids[ids==18446744073709551615] = 0
        ids[((uvs<5).any(axis=1))|(uvs>=np.array([imgwidth_new, imgheight_new])-5).any(axis=1)] = 0
        kpt_uv.append(np.hstack((uvs[ids>0], np.int32(ids[ids>0,None])-1)))
        
    kpts3d0 = np.stack([recon.points3D[i+1].xyz if i+1 in recon.points3D else np.array([1e8,1e8,1e8]) for i in range(np.uint32(list(recon.points3D.keys())).max())])
    kpt_d = []
    kpt_w = []
    kpt_id = []
    kpt_pairid = []
    for i, p in enumerate(pairs):
        d0 = data[i,kpt_uv[p[0]][:,1],kpt_uv[p[0]][:,0],:2]
        d1 = data[i,kpt_uv[p[1]][:,1],kpt_uv[p[1]][:,0],2:]
        kpt_d.append(np.hstack([d0[:,0],d1[:,0]]))
        kpt_w.append(np.hstack([d0[:,1],d1[:,1]]))
        kpt_id.append(np.hstack([kpt_uv[p[0]][:,2], kpt_uv[p[1]][:,2]]))
        kpt_pairid.append(np.stack((np.ones(len(d0)+len(d1),dtype=np.int32)*i, np.hstack((np.zeros(len(d0),dtype=np.int32), M*np.ones(len(d1),dtype=np.int32))))).T)
    kpt_d = np.hstack(kpt_d) 
    kpt_w = np.hstack(kpt_w)
    kpt_id = np.hstack(kpt_id)
    kpt_pairid = np.vstack(kpt_pairid)
    kpts3d = torch.from_numpy(np.hstack((kpts3d0, np.ones((len(kpts3d0),1))))).float().cuda()[kpt_id]
    kpt3d_gt = (RT0.inverse()[pairs[kpt_pairid[:,0],(kpt_pairid[:,1]>0).astype(np.int32)], 2]*kpts3d).sum(dim=1)
    kpt3d_d = torch.from_numpy(kpt_d).cuda()
    kpt_w = torch.from_numpy(kpt_w).cuda()
    kpt3d_d_pow = torch.stack((kpt3d_d**2,kpt3d_d**4)).T

    # exit()
    S = scale.exp().detach()[kpt_pairid[:,0]+kpt_pairid[:,1]]
    dpow = torch.nn.Parameter(torch.ones(2*M).cuda())
    ddelta = torch.nn.Parameter(torch.zeros(2*M).cuda())
    torch.set_grad_enabled(True)
    L = []
    optimizer = torch.optim.Adam([dpow, ddelta], lr=0.01)
    for i in tqdm(range(300)):
        optimizer.zero_grad()
        lr = cosine_schedule(i/300, 1e-2, 1e-4)
        adjust_learning_rate_by_lr(optimizer, lr)
        # loss = ((S*(torch.pow(kpt3d_d,dpow[kpt_pairid[:,0]+kpt_pairid[:,1]]))-kpt3d_gt).abs()*kpt_w).sum()/len(kpt3d_d)*10
        loss = ((S*(torch.pow(kpt3d_d,dpow[kpt_pairid[:,0]+kpt_pairid[:,1]])+ddelta[kpt_pairid[:,0]+kpt_pairid[:,1]]*kpt3d_d)-kpt3d_gt).abs()*kpt_w).sum()/len(kpt3d_d)*10
        L.append(loss.item())
        loss.backward()
        optimizer.step()
        
    optimizer = torch.optim.Adam([scale, focal_scale, scale1, dpow, ddelta], lr=0.001)
    L = []
    for i in tqdm(range(300)):
        optimizer.zero_grad()
        lr = cosine_schedule(i/300, 1e-3, 1e-5)
        adjust_learning_rate_by_lr(optimizer, lr)
        RT = torch.vstack((RT0[pairs[:,0]], RT0[pairs[:,1]]))
        S = scale.exp()#/(scale.mean()).exp()
        S1 = scale1.exp()
        SRT = RT*torch.stack((S*S1[0],S*S1[1],S,torch.ones_like(S))).T[:,None]
        r = ((SRT[:M]@x)[:,:-1]-(SRT[M:2*M]@y)[:,:-1])*w
        loss = (r.norm(dim=[1,2])).log1p().mean()
        loss += ((S1.T*focal_scale.exp())*norm_fs-1).abs().mean()*1000
        # loss += ((S[kpt_pairid[:,0]+kpt_pairid[:,1]]*(torch.pow(kpt3d_d,dpow[kpt_pairid[:,0]+kpt_pairid[:,1]]))-kpt3d_gt).abs()*kpt_w).sum()/len(kpt3d_d)*10
        loss += ((S[kpt_pairid[:,0]+kpt_pairid[:,1]]*(torch.pow(kpt3d_d,dpow[kpt_pairid[:,0]+kpt_pairid[:,1]])+kpt3d_d*ddelta[kpt_pairid[:,0]+kpt_pairid[:,1]])-kpt3d_gt).abs()*kpt_w).sum()/len(kpt3d_d)*10
        # loss += ((S[kpt_pairid[:,0]+kpt_pairid[:,1]]*torch.pow(kpt3d_d,dpow[kpt_pairid[:,0]+kpt_pairid[:,1]])*ddelta[kpt_pairid[:,0]+kpt_pairid[:,1]]-kpt3d_gt).abs()*kpt_w).sum()/len(kpt3d_d)*10
        # loss += ((S[kpt_pairid[:,0]+kpt_pairid[:,1]]*(torch.pow(kpt3d_d,dpow[kpt_pairid[:,0]+kpt_pairid[:,1]])*ddelta[kpt_pairid[:,0]+kpt_pairid[:,1],0]+kpt3d_d*ddelta[kpt_pairid[:,0]+kpt_pairid[:,1],1])-kpt3d_gt).abs()*kpt_w).sum()/len(kpt3d_d)*10
        # loss += ((S[kpt_pairid[:,0]+kpt_pairid[:,1]]*kpt3d_d*((dpow[kpt_pairid[:,0]+kpt_pairid[:,1]]*kpt3d_d_pow).sum(dim=1)+1)-kpt3d_gt).abs()*kpt_w).sum()/len(kpt3d_d)*10
        L.append(loss.item())
        loss.backward()
        optimizer.step()
        
    torch.set_grad_enabled(False)
    torch.cuda.empty_cache()
    depth0 = (torch.pow(data[...,0],dpow[:M,None,None].cpu())+data[...,0]*ddelta[:M,None,None].cpu())*S[:M,None,None].cpu()
    depth1 = (torch.pow(data[...,2],dpow[M:,None,None].cpu())+data[...,2]*ddelta[M:,None,None].cpu())*S[M:,None,None].cpu()
    # depth0 = data[...,0]*(1+data[...,0]**2*dpow[:M,0,None,None].cpu()+data[...,0]**4*dpow[:M,1,None,None].cpu())*S[:M,None,None].cpu()
    # depth1 = data[...,2]*(1+data[...,2]**2*dpow[M:,0,None,None].cpu()+data[...,2]**4*dpow[M:,1,None,None].cpu())*S[M:,None,None].cpu()
    depth = torch.concat((depth0[:,None], depth1[:,None]), dim=1)
    depth_img = torch.zeros(N,*imgshape[0])
    w_img = torch.zeros(N,*imgshape[0])
    w_img_max = torch.zeros(N,*imgshape[0])
    for k in tqdm(range(N)):
        i,j = torch.where(pairs==k)
        w_img[k] = (data[i,...,j*2+1]**2).sum(dim=0)
        depth_img[k] = (depth[i,j]*(data[i,...,j*2+1]**2)).sum(dim=0)/w_img[k]
        w_img[k] = w_img[k].sqrt()
        w_img_max[k] = data[i,...,j*2+1].max(dim=0)[0]
           
    pairs_name = np.array([imgfile[:-4] for imgfile in dataset.image_paths])[pairs]
    cx, cy = K[:2,2]
    grid = torch.from_numpy(np.stack(((mapx-cx)/cx, (mapy-cy)/cy)).transpose((1,2,0))[None])
    S = scale.exp().detach().cpu()
    data[...,0] *= S[:M,None,None]
    data[...,2] *= S[M:,None,None]
    torch.save((pairs_name, data, grid), path+'/../pairs_uni.pth')
 
    import open3d as o3d
    from tqdm import tqdm
    import numpy as np
    D, Dw = depth_img, w_img
    RT = RT0.cpu()
    errdth = (D.mean()*0.3).item()
    errwth = (w_img_max.mean()*0.3).item()
    kpt_need = []

    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(D.shape[2], D.shape[1], focal_scale.exp()[0].item(), focal_scale.exp()[1].item(), K_new[0,2], K_new[1,2])
    voxel_size = 0.05#(RT[:,:3,3]-RT[:,:3,3].mean(dim=0)).norm(dim=1).max()/100
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size, sdf_trunc=voxel_size*5,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    dth = np.percentile(D.ravel(), (5,95))[1]
    for i in tqdm(range(N)):
        depth = D[i].cpu().numpy().copy()
        d0 = (w2c[i,:3,:3]@kpts3d0[kpt_uv[i][:,2]].T+w2c[i,:3,3:])[2]
        d1 = depth[kpt_uv[i][:,1],kpt_uv[i][:,0]]
        dwpt = w_img_max[i,kpt_uv[i][:,1],kpt_uv[i][:,0]]
        depth[Dw[i].cpu().numpy()<min(Dw[i].mean().item()*0.1, 0.1)] = dth*10
        dmax = d1[(d0-d1>errdth)&(dwpt<errwth)]
        if dmax.any():
            m = (depth>np.median(dmax))&(w_img_max[i].numpy()<errwth)
            depth[m] = dth*10
            kpt_need.append(kpt_uv[i][m[kpt_uv[i][:,1],kpt_uv[i][:,0]],2])
        if np.isnan(depth).any(): continue
        color = np.uint8((images_undist[i].permute(1,2,0).numpy().copy()*0.5+0.5).clip(0,1)*255).copy()
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color), 
            o3d.geometry.Image(depth), 
            depth_scale=1., depth_trunc=dth, convert_rgb_to_intensity=False)
        pose = RT[i].inverse().cpu().numpy()
        volume.integrate(rgbd, intr, pose)
    mesh = volume.extract_triangle_mesh()
    pts = np.asarray(volume.extract_voxel_point_cloud().points)
    del volume
    mesh.compute_vertex_normals()
    
    kpt_need = np.hstack(kpt_need)
    kpt_need = np.unique(kpt_need)
    vert_need = kpts3d0[kpt_need]
    color_need = np.stack([recon.points3D[i+1].color if i+1 in recon.points3D else np.uint8([0,0,0]) for i in range(np.uint32(list(recon.points3D.keys())).max())])[kpt_need]
    # trimesh.PointCloud(np.vstack((mesh.vertices, vert_need)), np.vstack((mesh.vertex_colors, color_need/255))).export('mesh.ply');
    # o3d.io.write_triangle_mesh('mesh.ply', mesh)
    kdt = KDTree(np.asarray(mesh.vertices))
    d = kdt.query(vert_need, k=1)[0]
    trimesh.PointCloud(np.vstack((mesh.vertices, vert_need[d>errdth*0.3])), np.vstack((mesh.vertex_colors, color_need[d>errdth*0.3]/255))).export(path+'/../mesh.ply');
    
