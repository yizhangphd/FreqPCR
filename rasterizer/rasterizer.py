from . import _C
import torch

def get_bin_size(img_size):
    if img_size <= 256:
        return 16
    elif img_size <= 512:
        return 32
    elif img_size <= 1024:
        return 64
    else:
        return 128

def rasterize(xyz_ndc, hw, radius, buf_num=1):
    """
    This function implements rasterization.
    Args: 
        xyz_ndc: ndc coordinates of point cloud
        hw: height and width of rasterization
        radius: radius of points
    Output:
        idx: buffer of points index
        zbuf: buffer of points depth
    """

    N = xyz_ndc.size(0)
    points_per_pixel = buf_num
    bin_size = get_bin_size(hw[0])
    cloud_to_packed_first_idx = torch.tensor([0], device=xyz_ndc.device)
    num_points_per_cloud = torch.tensor([N], device=xyz_ndc.device)
    radius = radius * torch.ones([N], device=xyz_ndc.device)
    idx, zbuf, _ = _C._rasterize(xyz_ndc, cloud_to_packed_first_idx, num_points_per_cloud, hw, radius, points_per_pixel, bin_size, N)
    
    return idx, zbuf

def rasterize_multi_r(xyz_ndc, hw, r_list, n_list, buf_num=1):
    """
    This function implements rasterization.
    Args: 
        xyz_ndc: ndc coordinates of point cloud
        hw: height and width of rasterization
        radius: radius of points
    Output:
        idx: buffer of points index
        zbuf: buffer of points depth
    """

    N = xyz_ndc.size(0)
    points_per_pixel = buf_num
    bin_size = get_bin_size(hw[0])
    cloud_to_packed_first_idx = torch.tensor([0], device=xyz_ndc.device)
    num_points_per_cloud = torch.tensor([N], device=xyz_ndc.device)
    radius = []
    for i, n in enumerate(n_list):
        radius.append(r_list[i] * torch.ones([n], device=xyz_ndc.device))
    radius = torch.cat(radius, dim=0)
    idx, zbuf, _ = _C._rasterize(xyz_ndc, cloud_to_packed_first_idx, num_points_per_cloud, hw, radius, points_per_pixel, bin_size, N)
    
    return idx, zbuf