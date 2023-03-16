import torch
import torch.utils.data as data
import os
import numpy as np
import json
from PIL import Image
from .utils import load_pc, get_rays, PointCloud, load_pc_np, setup_scene, get_proj_matrix
from torchvision import transforms as T
import time

from .rasterizer import MultiscaleRender
from .shader import NNScene
from glumpy import app, gloo, gl

class nerfDataset(data.Dataset):

    def __init__(self, args, split, mode):
    
        self.img_wh = (args.W, args.H)
        self.device = args.device
        self.pc_dir = args.pcdir
        self.mode = mode

        datadir = args.datadir
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        with open(os.path.join(datadir, 'transforms_' + split + '.json'), 'r') as f:
            self.meta = json.load(f)
        self.transform = T.ToTensor()
        focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x']) 
        focal *= self.img_wh[0] / 800
        self.intrinsic = torch.tensor([[focal, 0, self.img_wh[0] / 2], [0, focal, self.img_wh[1] / 2], [0, 0, 1]], dtype=torch.float32)

        self.id_list = [i for i in range(len(self.meta["frames"]))]
        self.img_list = []
        self.w2c_list = []
        self.ray_list = []
        self.c2w_list = []

        for idx in self.id_list:
            frame = self.meta['frames'][idx] 
            image_path = os.path.join(datadir, f"{frame['file_path']}.png")
            # print(image_path)

        # load img
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img).permute(1,2,0)
            self.img_list.append(img[...,:3] * img[...,-1:] + (1 - img[...,-1:])) 
            

        # load pose
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            pose_gl = pose.copy()
            pose_gl[:,1:3] = (-1) * pose_gl[:,1:3] # for opengl
            self.c2w_list.append(pose_gl)
            c2w = torch.tensor(pose, dtype=torch.float32)

            # load ray
            if mode == 'render':
                ray = get_rays(args.H, args.W, self.intrinsic, c2w)
                self.ray_list.append(ray)
            else:
                self.ray_list.append(torch.ones([0]))

            pose = np.linalg.inv(pose)
            self.w2c_list.append(torch.tensor(pose, dtype=torch.float32))

        # opengl
        self.scene = NNScene()
        xyz, normal, color = load_pc_np(args.pcdir)
        setup_scene(self.scene, xyz, color, normal, c2w.numpy())
        app.Window(visible=False) # creates GL context
        self.renderer = MultiscaleRender(self.scene, [800, 800])
        self.proj_mat = get_proj_matrix(self.intrinsic.numpy(), [800, 800])

    def get_pc(self):
        pc_xyz = load_pc(self.pc_dir, self.device)  # n,3
        pc = PointCloud(pc_xyz, self.intrinsic, self.device, self.img_wh)
        return pc

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        """
        Returns:
            data dict {"img.rgb": rgb (H W C),
                       "img.mask": mask (H,W 1),
                       "camera_mat": camera_mat (4,4)
        """
        idx = idx % self.__len__()
        rgb = self.img_list[idx]
        w2c = self.w2c_list[idx]
        ray = self.ray_list[idx]
        c2w = self.c2w_list[idx]
        # ray = get_rays(800, 800, self.intrinsic, torch.tensor(c2w))
        t1 = time.time()
        depth = self.renderer.render(view_matrix=c2w, proj_matrix=self.proj_mat) # np 800 800 1
        t2 = time.time()
        depth[depth==0] = -1
        

        return {"idx": str(idx).rjust(3,'0'),
                "rgb": rgb.to(self.device), 
                "w2c": w2c.to(self.device),
                "ray": ray.to(self.device),
                "zbuf": torch.tensor(depth, device=self.device),
                't':t2-t1}
