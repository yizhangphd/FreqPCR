import os
import cv2
import time
import torch
from utils import config_parser, lr_decay, mse2psnr, visualize_pc
from dataset.utils import load_pc, PointCloud
from dataset.dataset import nerfDataset, TTDataset, ScanDataset, DTUDataset
from model.renderer import Renderer_pc_opt
import torch.optim as optim
from backup_utils import backup_terminal_outputs, backup_code, set_seed
from torch.utils.tensorboard import SummaryWriter
from run_rasterize import run_rasterize


parser = config_parser()
args = parser.parse_args()

set_seed(42)
log_name = time.strftime(f"%y%m%d-%H%M%S-{args.expname}-{args.H}")
back_path = os.path.join(args.logdir, log_name)
os.makedirs(back_path)
backup_terminal_outputs(back_path)
backup_code(back_path, ignored_in_current_folder=['logs_pc_opt','logs_edit','ckpt','data','.git','pytorch_rasterizer.egg-info','build','logs','__pycache__'])
print(back_path)
logger = SummaryWriter(back_path)
pc_path = os.path.join(back_path, 'pc')
os.makedirs(pc_path)


def filling_holes(args, train_set, train_buf, pc_xyz):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    num_per_hole = args.num_per_hole
    r = args.patch_r
    train_buf = train_buf[...,:1]
    pc_colors = torch.ones([pc_xyz.shape[0], 3], device=args.device)
    maxpool = torch.nn.MaxPool2d(3, stride=1, padding=1) # avoid filling edges

    it = 0
    sum_num = 0
    for batch in train_loader:
        it += 1

        idx = [int(id) for id in batch['idx']]
        # if args.dataset == 'nerf':
        alpha_gt = batch['alpha'] # 1 400 400 1
        # else:
        #     alpha_gt = batch['rgb'] # 1 400 400 3
        #     alpha_gt = torch.mean(alpha_gt, dim=-1, keepdim=True) # 1 400 400 1
        #     alpha_gt[alpha_gt<1] = 0
        #     alpha_gt = 1 - alpha_gt
        ray = batch['ray'] # 1 400 400 7
        zbuf = train_buf[idx].to(args.device) # 1 400 400 1
        zbuf_ = zbuf.clone()
        zbuf[zbuf<0] = 0
        zbuf[zbuf>0] = 1
        alpha_gt[alpha_gt<1] = 0 # for nerf drums
        alpha_gt = 1 - maxpool(maxpool(1. - alpha_gt.permute(0,3,1,2)).permute(0,2,3,1))  # ignore the edge problem

        res = zbuf - alpha_gt
        # alpha_gt[alpha_gt<1] = 0
        hole_mask = (res.squeeze(-1).squeeze(0) < 0).int() # 400 400
        hole_idx = torch.nonzero(hole_mask)  # n_holes, 2
        hole_num = hole_idx.shape[0]
        print(f'Find {hole_num} holes in it {it}')
        if hole_num > 0:
            for i in range(hole_num):
                xy = hole_idx[i]  # 2
                x_min = max(0,xy[0].item()-r)
                x_max = min(args.H-1,xy[0].item()+r)
                y_min = max(0,xy[1].item()-r)
                y_max = min(args.W-1,xy[1].item()+r)
                patch = zbuf_[0,x_min:x_max,y_min:y_max,0]  # 2r, 2r
                if (patch > 0).int().sum() < 1:
                    print('patch too small!')
                    r = r * 2
                    continue

                d_min = patch[patch>0].min()
                d_max = patch[patch>0].max()
                o = ray[0,0,0,:3] # 3
                dirs = ray[0,xy[0].item(),xy[1].item(),3:6]# n_hole 3
                cos = ray[0,xy[0].item(),xy[1].item(),6:7] # n_hole 3
                
                depth = torch.linspace(d_min, d_max, steps=num_per_hole) # 20
                new_xyz = (o + depth.reshape(num_per_hole,1).to(args.device) * dirs.reshape(1,3) / cos.reshape(1,1)).reshape(-1,3)

                # print(new_xyz.shape)
                pc_xyz = torch.cat([pc_xyz, new_xyz], dim=0) 
                adding_num = new_xyz.shape[0]
                new_color = torch.tensor([[1,0,0]], device=args.device).expand(adding_num, 3)
                pc_colors = torch.cat([pc_colors, new_color], dim=0) 
                # exit()
            sum_num += num_per_hole * hole_num
            if sum_num > args.max_fill_num:
                print('too many!! break!')
                break
    print(f'Done. adding num:{sum_num}')
    visualize_pc(pc_xyz, pc_colors, os.path.join(pc_path, time.strftime(f"{args.expname}-adding-%y%m%d-%H%M%S.ply")))
    del train_loader, train_buf
    
    return pc_xyz
    

if __name__ == '__main__':

    if args.dataset == 'tt':
        train_set = TTDataset(args, 'train', 'render')
        test_set = TTDataset(args, 'test', 'render', cut=1)
    elif args.dataset == 'nerf':
        train_set = nerfDataset(args, 'train', 'render')
        test_set = nerfDataset(args, 'test', 'render', cut=1)
    elif args.dataset == 'scan':
        train_set = ScanDataset(args, 'train', 'render')
        test_set = ScanDataset(args, 'test', 'render', cut=1)
    elif args.dataset == 'dtu':
        train_set = DTUDataset(args, 'train', 'render')
        test_set = DTUDataset(args, 'test', 'render', cut=1)
    else:
        assert False

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)

    pc = train_set.get_pc()
    pc_xyz = pc.xyz
    renderer = Renderer_pc_opt(args, pc_xyz.shape[0])
    
    # Optimizer
    # opt_para = []
    # opt_para.append({"params": renderer.mlp.parameters(), "lr": args.mlp_lr})  
    # opt_para.append({"params": renderer.sigma, "lr": args.sig_lr})  
    # opt = optim.Adam(opt_para)
    
    it = 0
    for loop in range(args.bigloop):
        epoch = 0
        print(time.strftime(f"[%H:%M:%S] Loop {loop} Begin."))

        if not args.denoise_only:
            test_buf, test_id_buf, train_buf, train_id_buf = run_rasterize(pc, test_set, train_set, args, args.buf)
            test_buf, test_id_buf, train_buf, train_id_buf = torch.tensor(test_buf), torch.tensor(test_id_buf), torch.tensor(train_buf), torch.tensor(train_id_buf)
            # print(train_id_buf.max())
            print(time.strftime(f"[%H:%M:%S] Rasterize Finish"))
            # torch.cuda.empty_cache()

            pc_xyz = filling_holes(args, train_set, train_buf, pc_xyz)
            pc = PointCloud(pc_xyz, train_set.intrinsic, train_set.device, train_set.img_wh)

        if args.reset:
            # reset mlp
            renderer = Renderer_pc_opt(args, pc_xyz.shape[0])
        else:
            # only reset sigma
            renderer.update_sigma(pc_xyz.shape[0])
        # reset opt
        opt_para = []
        opt_para.append({"params": renderer.mlp.parameters(), "lr": args.mlp_lr})  
        opt_para.append({"params": renderer.sigma, "lr": args.sig_lr})  
        opt = optim.Adam(opt_para)



        test_buf, test_id_buf, train_buf, train_id_buf = run_rasterize(pc, test_set, train_set, args, args.buf)
        test_buf, test_id_buf, train_buf, train_id_buf = torch.tensor(test_buf), torch.tensor(test_id_buf), torch.tensor(train_buf), torch.tensor(train_id_buf)
        del pc
        dep_max = train_buf.max()
        print(time.strftime(f"[%H:%M:%S] Rasterize Finish"))

        # denoise
        e = args.epoch

        for _ in range(e):
            renderer.train()
            epoch += 1
            for batch in train_loader:
                it += 1
                idx = [int(id) for id in batch['idx']]
                ray = batch['ray'] # torch.stack(batch['ray'], dim=0) # h w 7
                img_gt = batch['rgb'] #torch.stack(batch['rgb'], dim=0) # h w 3
                alpha_gt = batch['alpha']
                idbuf = train_id_buf[idx].to(args.device)
                zbuf = train_buf[idx].to(args.device) # h w 1

                img_pre, acc_pre, dep, gt = renderer(zbuf, ray, idbuf, True, torch.cat([img_gt, alpha_gt], dim=-1))
            
                opt.zero_grad()
                if args.denoise_only:
                    loss_alpha = torch.mean((acc_pre - gt[...,-1:])[gt[...,-1]==1] ** 2) #[gt[...,-1]==1] ** 2)
                    loss_l2 = torch.mean((img_pre - gt[...,:-1])[gt[...,-1]==1] ** 2) #[gt[...,-1]==1] ** 2)
                else:
                    loss_alpha = torch.mean((acc_pre - gt[...,-1:]) ** 2)
                    loss_l2 = torch.mean((img_pre - gt[...,:-1]) ** 2)
                sigma = torch.sigmoid(renderer.sigma)
                s = 0.99 * sigma + 0.005 # avoid overflow
                reg = torch.mean(torch.log(s) + torch.log(1 - s))
                loss = loss_l2 + args.reg1 * reg #+ args.reg2 * torch.mean(s)
                if args.dataset == 'nerf' or args.dataset == 'dtu':
                    loss += loss_alpha

                loss.backward()
                opt.step()
                if it % 20 == 0:
                    logger.add_histogram('sigma', sigma, it)
                if it % 200 == 0:
                    psnr = mse2psnr(loss_l2)
                    print('[{}]-it:{}, psnr:{:.4f}, l2_loss:{:.4f}, reg:{:.4f}, alpha:{:.4f}'.format(back_path, it, psnr.item(), loss_l2.item(), reg.item(), loss_alpha.item()))
                    logger.add_scalar('train/psnr', psnr.item(), it)

                if it % 100 == 0:
                    img_pre = torch.clamp(img_pre, 0, 1)
                    dep = dep.detach().cpu().numpy()
                    dep = dep[0]
                    dep[dep==0] = dep_max + 0.5
                    dep = (dep - dep.min()) / (dep.max() - dep.min() + 1e-6)
                    dep = cv2.applyColorMap(cv2.convertScaleAbs(dep * 255., alpha=1),cv2.COLORMAP_JET) / 255.
                    
                    logger.add_image('train/img_fine', img_pre[0], global_step=it, dataformats='HWC')
                    logger.add_image('train/gtimg', gt[...,:-1][0], global_step=it, dataformats='HWC')
                    logger.add_image('train/dep', dep, global_step=it, dataformats='HWC')
                # torch.cuda.empty_cache()
                lr_decay(opt, 0.9999)

            if epoch % args.test_freq == 0:
                # print(time.strftime(f"[%H:%M:%S] TEST BEGIN"))
                renderer.eval()
                with torch.autograd.no_grad():
                    for i, batch in enumerate(test_loader):
                        idx = [int(id) for id in batch['idx']]
                        ray = batch['ray'] # torch.stack(batch['ray'], dim=0) # h w 7
                        img_gt = batch['rgb'].permute(0,3,1,2) #torch.stack(batch['rgb'], dim=0) # h w 3
                        zbuf = test_buf[idx].to(args.device)
                        idbuf = test_id_buf[idx].to(args.device)
                        img_pre, acc_pre, dep, _ = renderer(zbuf, ray, idbuf)
                        img_pre = img_pre.permute(0,3,1,2)
                        img_pre = torch.clamp(img_pre, 0, 1)
                        logger.add_image('test/img', img_pre[0].permute(1,2,0), global_step=it, dataformats='HWC')
                        logger.add_image('test/acc', acc_pre[0], global_step=it, dataformats='HWC')
                        # logger.add_image('test/dep', dep, global_step=it, dataformats='HWC')
                        dep = dep.detach().cpu().numpy()
                        dep = dep[0]
                        dep[dep==0] = dep_max + 0.5
                        dep = (dep - dep.min()) / (dep.max() - dep.min())
                        dep = cv2.applyColorMap(cv2.convertScaleAbs(dep * 255., alpha=1),cv2.COLORMAP_JET) / 255.
                        logger.add_image('test/dep', dep, global_step=it, dataformats='HWC')


        sigma = torch.sigmoid(renderer.sigma) 
        mask = sigma[:,0] > args.thres
        pc_xyz = pc_xyz[mask]
        visualize_pc(pc_xyz, sigma.expand(-1,3)[mask], os.path.join(pc_path, time.strftime(f"{args.expname}-denoise-%y%m%d-%H%M%S.ply")))
        print(time.strftime(f"[%H:%M:%S] Denoise Finish"))
        pc = PointCloud(pc_xyz, train_set.intrinsic, train_set.device, train_set.img_wh)

        # if not args.denoise_only:
        #     test_buf, test_id_buf, train_buf, train_id_buf = run_rasterize(pc, test_set, train_set, args, args.buf)
        #     test_buf, test_id_buf, train_buf, train_id_buf = torch.tensor(test_buf), torch.tensor(test_id_buf), torch.tensor(train_buf), torch.tensor(train_id_buf)
        #     # print(train_id_buf.max())
        #     print(time.strftime(f"[%H:%M:%S] Rasterize Finish"))
        #     # torch.cuda.empty_cache()

        #     pc_xyz = filling_holes(args, train_set, train_buf, pc_xyz)
        #     pc = PointCloud(pc_xyz, train_set.intrinsic, train_set.device, train_set.img_wh)

        # if args.reset:
        #     # reset mlp
        #     renderer = Renderer_pc_opt(args, pc_xyz.shape[0])
        # else:
        #     # only reset sigma
        #     renderer.update_sigma(pc_xyz.shape[0])
        # # reset opt
        # opt_para = []
        # opt_para.append({"params": renderer.mlp.parameters(), "lr": args.mlp_lr})  
        # opt_para.append({"params": renderer.sigma, "lr": args.sig_lr})  
        # opt = optim.Adam(opt_para)

