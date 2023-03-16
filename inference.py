import os
import time
import torch
from utils import config_parser, load_fragments, mse2psnr
from dataset.dataset import nerfDataset
from model.renderer import Renderer
import matplotlib.pyplot as plt
from backup_utils import backup_terminal_outputs, backup_code, set_seed
from torch.utils.tensorboard import SummaryWriter
from piqa import PSNR

parser = config_parser()
args = parser.parse_args()

set_seed(1023)
back_path = os.path.join('logs', time.strftime(f"%y%m%d-%H%M%S-Infer-{args.expname}"))
os.makedirs(back_path)
backup_terminal_outputs(back_path)
backup_code(back_path, ignored_in_current_folder=['logs_opensource','logs_pc_opt','logs_edit','ckpt','data','.git','pytorch_rasterizer.egg-info','build','logs','__pycache__'])
print(back_path)
logger = SummaryWriter(back_path)
video_path = os.path.join(back_path, 'video')
os.makedirs(video_path)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    if args.dataset == 'nerf':
        test_set = nerfDataset(args, 'test', 'render')
    else:
        assert False
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)

    renderer = Renderer(args)
    edge = args.edge_mask
    _, test_buf = load_fragments(args)  # cpu 100 800 800 k
    renderer.load_state_dict(torch.load(args.ckpt))
    print('TEST BEGIN!!!')

    renderer.eval()
    PSNR = 0
    with torch.autograd.no_grad():
        for i, batch in enumerate(test_loader):
            idx = [int(id) for id in batch['idx']]
            ray = batch['ray'] # b h w 7
            img_gt = batch['rgb'].permute(0,3,1,2)  # b 3 h w
            zbuf = test_buf[idx].to(args.device) # b h w 1
            
            output = renderer(zbuf, ray, gt=None, mask_gt=None, isTrain=False)
            img_pre = output['img']
            img_pre = torch.clamp(img_pre, 0, 1)
            loss = torch.mean((img_gt - img_pre) ** 2)
            psnr = mse2psnr(loss)
            PSNR += psnr
            img_pre = img_pre.squeeze(0).permute(1,2,0).cpu().numpy()
            
            plt.imsave(os.path.join(video_path, str(i).rjust(3,'0') + '.png'), img_pre)
        print(f'Done. Save path:{back_path}, PSNR:{PSNR.item()/200}')
   