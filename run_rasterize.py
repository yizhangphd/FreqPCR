import torch
from rasterizer.rasterizer import rasterize
from utils import config_parser
from dataset.dataset import nerfDataset, ScanDataset, DTUDataset, TTDataset, toyDeskDataset
import os
import numpy as np
import time
from utils import config_parser
from tqdm import tqdm


def run_rasterize(pc, test_set, train_set, args, buf_num=1):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)
    z_list = []
    id_list = []
    for batch in tqdm(test_loader):
        pose = batch['w2c'][0]
        xyz_ndc = pc.get_ndc(pose)
        id, zbuf = rasterize(xyz_ndc, (args.H, args.W), args.radius, buf_num)
        z_list.append(zbuf.float().cpu())
        id_list.append(id.long().cpu())
        # if i % 20 == 0:
        #     print('test', i)
    z_test = torch.cat(z_list, dim=0).numpy() 
    id_test = torch.cat(id_list, dim=0).numpy()  
    # print('z_list.shape', z_list.shape)
    # np.save(test_z_path, z_list)
    # np.save(test_id_path, id_list)

    # train set 
    z_list = []
    id_list = []
    for batch in tqdm(train_loader):
        pose = batch['w2c'][0]
        xyz_ndc = pc.get_ndc(pose)
        id, zbuf = rasterize(xyz_ndc, (args.H, args.W), args.radius, buf_num)
        z_list.append(zbuf.float().cpu())
        id_list.append(id.long().cpu())
        # if i % 20 == 0:
        #     print('train', i)
    z_train = torch.cat(z_list, dim=0).numpy() 
    id_train = torch.cat(id_list, dim=0).numpy() 
    del train_loader, test_loader, z_list, id_list, id, zbuf
    return z_test, id_test, z_train, id_train


if __name__ == '__main__':

    parser = config_parser()
    args = parser.parse_args()

    if args.dataset == 'scan':
        train_set = ScanDataset(args, 'train', 'rasterize')
        test_set = ScanDataset(args, 'test', 'rasterize')
    elif args.dataset == 'nerf':
        train_set = nerfDataset(args, 'train', 'rasterize')
        test_set = nerfDataset(args, 'test', 'rasterize')
    elif args.dataset == 'dtu':
        train_set = DTUDataset(args, 'train', 'rasterize')
        test_set = DTUDataset(args, 'test', 'rasterize')
    elif args.dataset == 'tt':
        train_set = TTDataset(args, 'train', 'render')
        test_set = TTDataset(args, 'test', 'render')
    elif args.dataset == 'toy':
        train_set = toyDeskDataset(args, 'train', 'render')
        test_set = toyDeskDataset(args, 'test', 'render')
    else:
        assert False

    

    if not os.path.exists(args.frag_path):
        os.makedirs(args.frag_path)

    
    test_id_path = str(args.radius) + '-idx-' + str(args.H) + '-test.npy'
    test_z_path = str(args.radius) + '-z-' + str(args.H) + '-test.npy'
    test_id_path = os.path.join(args.frag_path, test_id_path) 
    test_z_path = os.path.join(args.frag_path, test_z_path) 

    train_id_path = str(args.radius) + '-idx-' + str(args.H) + '-train.npy'
    train_z_path = str(args.radius) + '-z-' + str(args.H) + '-train.npy'
    train_id_path = os.path.join(args.frag_path, train_id_path) 
    train_z_path = os.path.join(args.frag_path, train_z_path) 
    
    # begin = time.time()

    pc = train_set.get_pc()
    z_test, id_test, z_train, id_train = run_rasterize(pc, test_set, train_set, args)
    # test set 
    # z_list = []
    # id_list = []
    # for i, batch in enumerate(test_loader):
    #     pose = batch['w2c'][0]
    #     xyz_ndc = pc.get_ndc(pose)
    #     id, zbuf = rasterize(xyz_ndc, (args.H, args.W), args.radius)
    #     z_list.append(zbuf.float().cpu())
    #     id_list.append(id.long().cpu())
    #     if i % 20 == 0:
    #         print('test', i)
    # z_list = torch.cat(z_list, dim=0).numpy() 
    # id_list = torch.cat(id_list, dim=0).numpy()  
    # print('z_list.shape', z_list.shape)
    np.save(test_z_path, z_test)
    np.save(test_id_path, id_test)

    # # train set 
    # z_list = []
    # id_list = []
    # for i, batch in enumerate(train_loader):
    #     pose = batch['w2c'][0]
    #     xyz_ndc = pc.get_ndc(pose)
    #     id, zbuf = rasterize(xyz_ndc, (args.H, args.W), args.radius)
    #     z_list.append(zbuf.float().cpu())
    #     id_list.append(id.long().cpu())
    #     if i % 20 == 0:
    #         print('train', i)
    # z_list = torch.cat(z_list, dim=0).numpy() 
    # id_list = torch.cat(id_list, dim=0).numpy() 
    # print('z_list.shape', z_list.shape)  
    np.save(train_z_path, z_train)
    np.save(train_id_path, id_train)
    

