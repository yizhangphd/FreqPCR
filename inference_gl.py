import os
import time
import torch
from utils import config_parser
from dataset_gl.dataset import nerfDataset
from model.renderer import Renderer
import matplotlib.pyplot as plt
from backup_utils import backup_terminal_outputs, backup_code, set_seed
from torch.utils.tensorboard import SummaryWriter

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

    test_set = nerfDataset(args, 'test', 'render')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)
    
    renderer = Renderer(args)
    if args.ckpt is not None:
        renderer.load_state_dict(torch.load(args.ckpt))
    edge = args.edge_mask

    

    print('TEST BEGIN!!!')
    renderer.eval()
    timeall = 0
    mlp = 0
    u = 0
    with torch.autograd.no_grad():
        
        begin = time.time()
        t_sum = 0
        for i, batch in enumerate(test_loader):
            # print(i)
            # idx = int(batch['idx'][0])
            ray = batch['ray']
            # img_gt = batch['rgb'][0]
            zbuf = batch['zbuf']
            output = renderer(zbuf, ray, gt=None, mask_gt=None, isTrain=False)
            if i > 0:
                timeall =  timeall + output['infer_time']
                mlp =  mlp + output['mlp_t']
                u =  u + output['u_t']
            # img_pre = output['img'].cpu().numpy()

            # img_pre[img_pre>1] = 1.
            # img_pre[img_pre<0] = 0.
            # plt.imsave(os.path.join('test_render', str(i).rjust(3,'0') + '.png'), img_pre)
            t_sum += batch['t'][0]
        end = time.time()
    print('time consume:', end - begin, 'feamap', t_sum)
    print(f'Done. {mlp/199}, {u/199}, {timeall/199} s')

                