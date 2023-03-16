from .net import AFNet, UNet, MLP_pc_opt
import torch
import torch.nn as nn
from torchvision import transforms as T
import time

class Renderer(nn.Module):
    """
    This class implements radiance mapping and refinement.
    """

    def __init__(self, args):
        super(Renderer, self).__init__()
        self.afnet = AFNet(args.dim).to(args.device)
        self.unet = UNet(args).to(args.device)
        self.dim = args.dim
        self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(args.scale_min, args.scale_max), ratio=(0.75, 1.25), interpolation=T.InterpolationMode.NEAREST)

        self.pad_w = T.Pad(args.pad, 1., 'constant')
        self.pad_b = T.Pad(args.pad, -1., 'constant')
        self.train_size = args.train_size

    def forward(self, zbuf, ray, gt, mask_gt, isTrain):

        if isTrain:
            
            ray = self.pad_w(ray.permute(0,3,1,2)) # b 7 H W
            gt = self.pad_w(gt.permute(0,3,1,2)) # b 3 H W
            zbuf = self.pad_b(zbuf.permute(0,3,1,2))

            if mask_gt is not None:
                # never pad
                mask_gt = mask_gt.permute(0,3,1,2)
                cat_img = torch.cat([ray, gt, zbuf, mask_gt], dim=1) 
            else:
                cat_img = torch.cat([ray, gt, zbuf], dim=1) 

            cat_img = self.randomcrop(cat_img)

            B, _, H, W = cat_img.shape

            ray = cat_img[:,:7].permute(0,2,3,1)  # b h w 3
            # cos = cat_img[:,3:4].permute(0,2,3,1) # b h w 1
            gt = cat_img[:,7:10] # b 3 h w
            zbuf = cat_img[:,10:11].permute(0,2,3,1)
            if mask_gt is not None:
                mask_gt = cat_img[:,11:12] # b h w 1

            pix_mask = (zbuf > 0).squeeze(-1)  # b h w 
            
        else:
            # print(zbuf.shape)
            # print(ray.shape)

            B, H, W, _ = zbuf.shape  
            pix_mask = (zbuf > 0).squeeze(-1)   # b h w  

        # o = o[pix_mask] # occ_point 3
        # dirs = dirs[pix_mask]  # occ_point 3
        # cos = cos[pix_mask]  # occ_point 1
        ray = ray[pix_mask] # occ_point 7
        zbuf = zbuf[pix_mask]  # occ_point 1
        
        xyz_near = ray[:,:3] + ray[:,3:6] * zbuf / ray[:,-1:] # occ_point 3
        
        feature = self.afnet(xyz_near, ray[:,3:6]) # occ_point 3
        

        feature_map = torch.ones([B, H, W, self.dim], device=xyz_near.device)
        feature_map[pix_mask] = feature
     
        
        feature_map = self.unet(feature_map[...,:-3].permute(0,3,1,2)) + feature_map[...,-3:].permute(0,3,1,2)
        

        return {'img':feature_map, 'gt':gt}


class EditRenderer(nn.Module):

    def __init__(self, args):
        super(EditRenderer, self).__init__()
        self.afnet = AFNet(args.dim).to(args.device)
        self.unet = UNet(args).to(args.device)
        self.dim = args.dim
        

    def forward(self, zbuf, ray, idbuf, edit_mask_list, T_list, scale=1.):
        
        B, H, W, _ = zbuf.shape  
        pix_mask = (zbuf > 0).squeeze(-1)   # b h w  
        idbuf = idbuf[pix_mask].squeeze(-1) # n


        ray = ray[pix_mask] # occ_point 7
        zbuf = zbuf[pix_mask]  # occ_point 1
        
        xyz_near = ray[:,:3] + ray[:,3:6] * zbuf / ray[:,-1:] # occ_point 3
        
        
        if len(edit_mask_list) > 0:
            if edit_mask_list[0] == 'all':
                T = T_list[0].to(xyz_near.device)
                xyz_near = (xyz_near - T[:3,-1:].t()) @ T[:3,:3] 
            else:
                
                for i, mask in enumerate(edit_mask_list):
                    mask_view = mask[idbuf.long()]
                    T = T_list[i].to(xyz_near.device)
                    xyz_near[mask_view] = (xyz_near[mask_view] - T[:3,-1:].t()) @ T[:3,:3] 
                    # xyz_near[mask_view] = (xyz_near[mask_view] - T1[:3,-1:].t()) / 1.5 - T0[:3,-1:].t()

        if scale != 1:
            xyz_near = xyz_near / scale
        
        feature = self.afnet(xyz_near, ray[:,3:6]) # occ_point 3
        
        feature_map = torch.ones([B, H, W, self.dim], device=xyz_near.device)
        feature_map[pix_mask] = feature
     
        
        feature_map = self.unet(feature_map[...,:-3].permute(0,3,1,2)) + feature_map[...,-3:].permute(0,3,1,2)
        

        return {'img':feature_map}


class Renderer_pc_opt(nn.Module):

    def __init__(self, args, points_num):
        super(Renderer_pc_opt, self).__init__()
        self.mlp = MLP_pc_opt(3).to(args.device)
        self.sigma = nn.Parameter(torch.zeros([points_num, 1], device=args.device))
        self.cropper = T.RandomCrop(size=args.train_size)
        self.ts = args.train_size
        self.args = args
        # self.maxpool = nn.MaxPool2d(3, stride=1, padding=1)


    def update_sigma(self, points_num):
        self.sigma = nn.Parameter(torch.zeros([points_num, 1], device=self.args.device))

    def forward(self, zbuf, ray, idx, isTrain=False, img_gt=None):
        """
        zbuf b 400 400 k
        ray b 400 400 7
        idx b 400 400 k
        img_gt b h w 4
        """
        B, H, W, K = idx.shape
        if isTrain:
            to_crop = torch.cat([zbuf, ray, idx, img_gt], dim=-1).permute(0,3,1,2) #b c h w
            to_crop = self.cropper(to_crop).permute(0,2,3,1)
            zbuf = to_crop[...,:K]
            ray = to_crop[...,K:K+7]
            idx = to_crop[...,K+7:2*K+7].long()
            img_gt = to_crop[...,2*K+7:2*K+11]
            H, W = self.ts, self.ts


        ray = ray.unsqueeze(-2).expand(B, H, W, K, 7)
        point_mask = zbuf > 0 # b h w k
        # pix_mask = 1 - self.maxpool(1. - (zbuf[..., 0] > 0).int().unsqueeze(1)).permute(0,2,3,1)  # b h w 1

        ray = ray[point_mask] # occ_point 7
        zbuf_ = zbuf[point_mask].reshape(-1,1)  # occ_point 1

        o = ray[...,:3] # occ_point 3
        dirs = ray[...,3:6] # occ_point 3
        cos = ray[...,-1:] # occ_point 1
        
        xyz_near = o + dirs * zbuf_ / cos # occ_point 3
        del o, cos, zbuf_
        colors = self.mlp(xyz_near, dirs) # occ_point 3

        color_map = torch.zeros([B, H, W, K, 3], device=xyz_near.device)
        color_map[point_mask] = colors
        sigma = torch.sigmoid(self.sigma)
        sigma_map = sigma[idx] # B, H, W, K, 1
        sigma_map[~point_mask] = 0

        sigma_map = sigma_map.reshape(B*H*W, K)
        color_map = color_map.reshape(B*H*W, K, 3)
        weights = sigma_map * torch.cumprod(torch.cat([torch.ones((sigma_map.shape[0], 1), device=xyz_near.device), 1.-sigma_map + 1e-10], -1), -1)[:, :-1]
        color_map = torch.sum(weights[...,None] * color_map, -2)  

        # depth_map = depth_map.reshape(B, H, W, 1)[:1]
        # depth_map[depth_map==0] = depth_map.max() + 0.5
        # depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

        depth_map = torch.sum(weights * zbuf.reshape(B*H*W, K), -1)
        depth_map = (depth_map - 2.) / (depth_map.max() - 2.)  # only for visualize
        acc_map = torch.sum(weights, -1)

        color_map = color_map + (1.-acc_map[...,None])

        # depth_map = cv2.applyColorMap(cv2.convertScaleAbs(depth_map[0].detach().cpu().numpy()*255., alpha=1), cv2.COLORMAP_JET) / 255.
        return color_map.reshape(B, H, W, 3), acc_map.reshape(B, H, W, 1), depth_map.reshape(B, H, W, 1), img_gt