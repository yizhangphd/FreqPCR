import torch
import numpy as np
import open3d as o3d
import time

class PointCloud():

    def __init__(self, xyz, intrinsic, device, img_wh):
        self.xyz = xyz
        self.intrinsic = intrinsic.to(device)
        self.device = device
        self.img_wh = img_wh

    def get_ndc(self, pose):

        K = self.intrinsic

        H = self.img_wh[1]
        W = self.img_wh[0]

        xyz_world = self.xyz # n 3
        n = xyz_world.size(0)
        pad = torch.ones([n, 1], device=self.device) # (N 1)
        xyz_world = torch.cat([xyz_world, pad], dim=1) # (N 4)

        xyz_cam = xyz_world @ pose.transpose(0,1) # (N 4)
        xyz_cam = xyz_cam[...,:3] # (N 3) 
        
        
        xyz_pix = xyz_cam @ K.transpose(0,1)  # n 3

        z_ndc = xyz_pix[...,2].unsqueeze(1) #  n 1
        xyz_pix = xyz_pix / (z_ndc.expand(n, 3))

        x_pix = xyz_pix[...,0].unsqueeze(1)
        x_ndc = 1 - (2 * x_pix) / (W - 1)
        y_pix = xyz_pix[...,1].unsqueeze(1)
        y_ndc = 1 - (2 * y_pix) / (H - 1)
        pts_ndc = torch.cat([x_ndc, y_ndc, z_ndc], dim=1)
        return pts_ndc


def load_pc(path, device, down=1):
    
    pc = o3d.io.read_point_cloud(path)
    if down > 1:
        pc = o3d.geometry.PointCloud.uniform_down_sample(pc, down)
    point = torch.tensor(np.asarray(pc.points), device=device, dtype=torch.float32)
    print('Load point cloud from {}, Point num={}, Down sample={}'.format(path, point.size(0), down))
    return point

def load_pc_np(path, down=5):
    
    pc = o3d.io.read_point_cloud(path)
    if down > 1:
        pc = o3d.geometry.PointCloud.uniform_down_sample(pc, down)
    xyz = np.asarray(pc.points)
    normal = np.asarray(pc.normals)
    color = np.asarray(pc.colors)
    print('Load point cloud from {}, Point num={}, Down sample={}'.format(path, xyz.shape[0], down))
    return xyz, normal, color

def get_rays(H, W, K, c2w):
        """
        generate ray directions
        """
        
        c2w = c2w.to('cuda:0') # TODO memory??
        rot = c2w[:3,:3] # 3 3
        trans = c2w[:3,3] # 3 3
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy') # H*W, H*W
        i = torch.tensor(i, device=c2w.device)
        j = torch.tensor(j, device=c2w.device)
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)  # H, W, 3

        norm_dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        cos = norm_dirs[:,:,-1:].float() # H, W
        # t1 = time.time()
        rays_d = torch.sum(dirs.reshape(H,W,1,3) * rot.reshape(1, 1, 3, 3), -1) 
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        rays_o = trans.reshape(1, 1, 3).expand(H, W, 3).float()
        # t2 = time.time()
        # print('ray' ,t2-t1)
        
        return torch.cat([rays_o, rays_d.float(), cos], dim=-1)



def setup_scene(scene, xyz, rgb, normals, view_mat):
    N = xyz.shape[0]
    xyz = xyz.astype(np.float32)
    rgb = rgb.astype(np.float32)
    normals = normals.astype(np.float32)
    print(xyz.shape, rgb.shape, normals.shape)

    scene.set_vertices(   #(395610, 3) (395610, 3) (395610, 3) (395610,) (395610, 2)
        positions=xyz,
        colors=rgb,
        normals=normals,
        uv1d=np.arange(N), # range(0,n)
        uv2d=np.zeros([N, 2]),  #zeros
        texture=None) # None

    scene.set_camera_view(view_mat)

    scene.set_model_view(np.eye(4))
    scene.set_indices(np.arange(3))

#     (4, 4) (3,) [[1. 0. 0. 0.]                                                                                                                                                          
#  [0. 1. 0. 0.]                                                                                                                                                                      
#  [0. 0. 1. 0.]                                                                                                                                                                      
#  [0. 0. 0. 1.]] [0 1 2] 


    scene.set_use_texture(False)

def get_proj_matrix(K, image_size, znear=.01, zfar=1000.):
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    width, height = image_size
    m = np.zeros((4, 4))
    m[0][0] = 2.0 * fx / width
    m[0][1] = 0.0
    m[0][2] = 0.0
    m[0][3] = 0.0

    m[1][0] = 0.0
    m[1][1] = 2.0 * fy / height
    m[1][2] = 0.0
    m[1][3] = 0.0

    m[2][0] = 1.0 - 2.0 * cx / width
    m[2][1] = 2.0 * cy / height - 1.0
    m[2][2] = (zfar + znear) / (znear - zfar)
    m[2][3] = -1.0

    m[3][0] = 0.0
    m[3][1] = 0.0
    m[3][2] = 2.0 * zfar * znear / (znear - zfar)
    m[3][3] = 0.0

    return m.T

