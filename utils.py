import configargparse
import os
import numpy as np
import torch
import cv2
import open3d as o3d

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))

def config_parser():

    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True)
    parser.add_argument("--expname", type=str, help='the name of experiment')
    parser.add_argument("--logdir", type=str, help='log directory')
    parser.add_argument("--datadir", type=str, help='data directory')
    parser.add_argument("--pcdir", type=str, help='point cloud directory')


    parser.add_argument("--radius", type=float, help='the radius of points when rasterizing')
    parser.add_argument("--frag_path", type=str, help='directory of saving fragments')
    parser.add_argument("--H", type=int)
    parser.add_argument("--W", type=int)
    parser.add_argument("--train_size", type=int, help='window size of training')
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--device", default='cuda:0', type=str)
    parser.add_argument("--scale_min", type=float, help='the minimum area ratio when random resize and crop')
    parser.add_argument("--scale_max", type=float, help='the maximum area ratio when random resize and crop')
    parser.add_argument("--thres", type=float, help='the maximum area ratio when random resize and crop')
    

    # parser.add_argument("--group_num", type=int)
    parser.add_argument("--dim", type=int, help='feature dimension of radiance mapping output')
    parser.add_argument("--u_lr", type=float, help='learning rate of unet')
    
    parser.add_argument("--mlp_lr", type=float, help='learning rate of mlp')
    parser.add_argument("--xyznear", action='store_true', default=False, help='corrdinates rectification or not')
    parser.add_argument("--pix_mask", action='store_true', default=False, help='using pixel mask or not')
    parser.add_argument("--U", type=int, help='down sampling times of unet')
    parser.add_argument("--udim", type=str, help='layers dimension of unet')
    parser.add_argument("--vgg_l", type=float, help='the weight of perceptual loss')
    parser.add_argument("--edge_mask", default=0, type=int, help='used in ScanNet 0000_00')
    parser.add_argument("--test_freq", default=10, type=int)
    parser.add_argument("--vid_freq", default=10, type=int)
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--bigloop", default=1, type=int)
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--pad", type=int, help='num of padding')
    parser.add_argument("--buf", type=int, help='num of padding')
    
    parser.add_argument("--denoise_only", action='store_true', default=False, help='corrdinates rectification or not')
    parser.add_argument("--reset", action='store_true', default=False, help='corrdinates rectification or not')
    parser.add_argument("--reg1", type=float)
    parser.add_argument("--reg2", type=float)
    parser.add_argument("--patch_r", type=int)
    parser.add_argument("--num_per_hole", type=int)
    parser.add_argument("--sig_lr", type=float, help='learning rate of unet')
    parser.add_argument("--max_fill_num", type=int)
    parser.add_argument("--down", default=1, type=int)
    parser.add_argument("--edit_mask_dir", type=str, help='layers dimension of unet')
    parser.add_argument("--afac", action='store_true', default=False, help='corrdinates rectification or not')
    
    
    return parser


def load_fragments(args):
    train_name = str(args.radius) + '-z-' + str(args.H) + '-train.npy'
    test_name = str(args.radius) + '-z-' + str(args.H) + '-test.npy'
    train_path = os.path.join(args.frag_path, train_name) 
    test_path = os.path.join(args.frag_path, test_name)
    train_buf = np.load(train_path)
    test_buf = np.load(test_path)
    print('Load fragments from', train_name, test_name)
    return torch.tensor(train_buf), torch.tensor(test_buf)

def load_idx(args):
    train_name = str(args.radius) + '-idx-' + str(args.H) + '-train.npy'
    test_name = str(args.radius) + '-idx-' + str(args.H) + '-test.npy'
    train_path = os.path.join(args.frag_path, train_name) 
    test_path = os.path.join(args.frag_path, test_name)
    train_buf = np.load(train_path)
    test_buf = np.load(test_path)
    print('Load fragments from', train_name, test_name)
    return torch.tensor(train_buf), torch.tensor(test_buf)

def lr_decay(opt, r=0.999):
    for p in opt.param_groups:
        p['lr'] = p['lr'] * r

def write_video(path, savepath, size):
    file_list = sorted(os.listdir(path))
    fps = 20
    four_cc = cv2.VideoWriter_fourcc(*'MJPG')
    save_path = savepath
    video_writer = cv2.VideoWriter(save_path, four_cc, float(fps), size)
    for item in file_list:
        if item.endswith('.jpg') or item.endswith('.png'):
            item = path + '/' + item
            img = cv2.imread(item)
            video_writer.write(img)

    video_writer.release()
    cv2.destroyAllWindows()


def visualize_pc(xyz, color, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(color.detach().cpu().numpy())
    o3d.io.write_point_cloud(path, pcd)
