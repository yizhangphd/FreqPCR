import os
import time
import torch
from utils import config_parser, load_fragments, load_idx, lr_decay, write_video, mse2psnr
from dataset.dataset import nerfDataset, ScanDataset, DTUDataset, TTDataset, toyDeskDataset
from model.renderer import Renderer
import matplotlib.pyplot as plt
import torch.optim as optim
from backup_utils import backup_terminal_outputs, backup_code, set_seed
from torch.utils.tensorboard import SummaryWriter
from piqa import SSIM, PSNR
import lpips

parser = config_parser()
args = parser.parse_args()

set_seed(1023)
back_path = os.path.join('logs', time.strftime("%y%m%d-%H%M%S-" + f'{args.expname}'))
os.makedirs(back_path)
backup_terminal_outputs(back_path)
backup_code(back_path, ignored_in_current_folder=['logs_pc_opt','logs_edit','ckpt','data','.git','pytorch_rasterizer.egg-info','build','logs','__pycache__'])
print(back_path)
logger = SummaryWriter(back_path)
video_path = os.path.join(back_path, 'video')
os.makedirs(video_path)


if __name__ == '__main__':

    if args.dataset == 'nerf':
        train_set = nerfDataset(args, 'train', 'render')
        test_set = nerfDataset(args, 'test', 'render')
    elif args.dataset == 'scan':
        train_set = ScanDataset(args, 'train', 'render')
        test_set = ScanDataset(args, 'test', 'render')
    elif args.dataset == 'dtu':
        train_set = DTUDataset(args, 'train', 'render')
        test_set = DTUDataset(args, 'test', 'render')
    elif args.dataset == 'tt':
        train_set = TTDataset(args, 'train', 'render')
        test_set = TTDataset(args, 'test', 'render')
    elif args.dataset == 'toy':
        train_set = toyDeskDataset(args, 'train', 'render')
        test_set = toyDeskDataset(args, 'test', 'render')
    else:
        assert False

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)
    
    renderer = Renderer(args)
    edge = args.edge_mask
    
    # Optimizer
    opt_para = []
    opt_para.append({"params": renderer.unet.parameters(), "lr": args.u_lr})  
    opt_para.append({"params": renderer.afnet.parameters(), "lr": args.mlp_lr}) 
    opt = optim.Adam(opt_para)

    fn_psnr = PSNR().to(args.device)
    # fn_lpips = LPIPS('vgg').to(args.device)
    loss_lpips = lpips.LPIPS(net='vgg').to(args.device)
    fn_ssim = SSIM().to(args.device)
    
    
    train_buf, test_buf = load_fragments(args)  # cpu 100 800 800 k
    if args.ckpt is not None:
        print(f'load model from {args.ckpt}')
        renderer.load_state_dict(torch.load(args.ckpt))
    
    it = 0
    epoch = 0
    best_psnr = 0
    training_time = 0.

    while True:
        
        # if epoch in [0, 2, 9, 24, 70, 140] or epoch % 300 == 0:
        if epoch % args.test_freq == 0:
            print('TEST BEGIN!!!')
            # video_it_path = os.path.join(video_path, str(epoch))
            # os.makedirs(video_it_path)

            test_psnr = 0
            test_lpips = 0
            test_ssim = 0

            renderer.eval()
            with torch.autograd.no_grad():
                for i, batch in enumerate(test_loader):
                    idx = [int(id) for id in batch['idx']]
                    ray = batch['ray'] # b h w 7
                    img_gt = batch['rgb'].permute(0,3,1,2)  # b 3 h w
                    zbuf = test_buf[idx].to(args.device) # b h w 1

                    output = renderer(zbuf, ray, gt=None, mask_gt=None, isTrain=False)
                    img_pre = torch.clamp(output['img'], 0, 1)
              
                    if edge > 0:
                        # for ScanNet_0000, since the artifacts at edges.
                        psnr = fn_psnr(img_pre[...,edge:-edge,edge:-edge], img_gt[...,edge:-edge,edge:-edge])
                        ssim = fn_ssim(img_pre[...,edge:-edge,edge:-edge], img_gt[...,edge:-edge,edge:-edge])
                        lpips_ = loss_lpips(img_pre[...,edge:-edge,edge:-edge], img_gt[...,edge:-edge,edge:-edge], normalize=True)
                    else:
                        psnr = fn_psnr(img_pre, img_gt)
                        ssim = fn_ssim(img_pre, img_gt)
                        lpips_ = loss_lpips(img_pre, img_gt, normalize=True)
                    test_lpips += lpips_.item()
                    test_psnr += psnr.item()
                    test_ssim += ssim.item()

                    # if epoch % args.vid_freq == 0: 

                    img_pre = img_pre.squeeze(0).permute(1,2,0)
                    img_pre = img_pre.cpu().numpy()
                    plt.imsave(os.path.join(video_path, str(i).rjust(3,'0') + '.png'), img_pre)
                    # torch.cuda.empty_cache()
                    
            test_lpips = test_lpips / len(test_set)
            test_psnr = test_psnr / len(test_set)
            test_ssim = test_ssim / len(test_set)
            logger.add_scalar('test/psnr', test_psnr, it)
            logger.add_scalar('test/lpips', test_lpips, it)
            logger.add_scalar('test/ssim', test_ssim, it)

            if test_psnr > best_psnr:
                best_psnr = test_psnr
                ckpt = os.path.join(back_path, 'model.pkl')
                torch.save(renderer.state_dict(), ckpt)
                print(f'Model Saved! Best PSNR: {best_psnr:{4}.{4}}')
            print(f'Test PSNR! Epoch:{epoch} Training_time: {training_time:{4}.{4}} hours, current: {test_psnr:{4}.{4}}, best: {best_psnr:{4}.{4}}')



        renderer.train()
        t1 = time.time()
        epoch += 1
        for batch in train_loader:
            it += 1
            idx = [int(id) for id in batch['idx']]
            ray = batch['ray'] # b h w 7
            img_gt = batch['rgb'] # b h w 3
            zbuf = train_buf[idx].to(args.device) # b h w 1
            # if args.dataset == 'dtu':
            #     mask_gt = batch['mask'] # b h w 1
            # else:
            # mask_gt = None

            output = renderer(zbuf, ray, img_gt, mask_gt=None, isTrain=True)

          
            img_pre = output['img']

            if output['gt'].min() == 1:
                # print('None img, skip')
                # torch.cuda.empty_cache()
                continue

            opt.zero_grad()
            loss_l2 = torch.mean((img_pre - output['gt']) ** 2)

            if args.vgg_l > 0:
                loss_vgg = torch.mean(loss_lpips(img_pre, output['gt'], normalize=True))
                loss = loss_l2 + args.vgg_l * loss_vgg
            else:
                loss = loss_l2 

            loss.backward()
            opt.step()

            if it % 100 == 0:
                psnr = mse2psnr(loss_l2)
                logger.add_scalar('train/psnr', psnr.item(), it)

            if it % 400 == 0:
                if args.vgg_l > 0:
                    print('[{}]-it:{}, psnr:{:.4f}, l2_loss:{:.4f}, vgg_loss:{:.4f}'.format(back_path, it, psnr.item(), loss_l2.item(), loss_vgg.item()))
                else:
                    print('[{}]-it:{}, psnr:{:.4f}, l2_loss:{:.4f}'.format(back_path, it, psnr.item(), loss.item()))
                img_pre[img_pre>1] = 1.
                img_pre[img_pre<0] = 0.
                logger.add_image('train/img_fine', img_pre[0].permute(1,2,0), global_step=it, dataformats='HWC')
            # torch.cuda.empty_cache()
        lr_decay(opt)
        t2 = time.time()
        training_time += (t2 - t1) / 3600
        


        