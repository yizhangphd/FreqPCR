import torch.nn as nn
import torch

def positional_encoding(tensor, num_encoding_functions=6, include_input=False, log_sampling=True, factor=1):
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions * factor,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq * 3.1415926))  

    # Special case, for no positional encoding
    return torch.cat(encoding, dim=-1)


class AFNet(nn.Module):

    def __init__(self, dim):
        super(AFNet, self).__init__()

        self.l1 = nn.Linear(120,256)
        self.l2 = nn.Linear(256,256) 
        self.l3 = nn.Linear(256,256) 
        self.l4 = nn.Linear(280,128)
        self.l5 = nn.Linear(128,dim)

        # self.ac = nn.ReLU()

        self.hyper = nn.Sequential(
            nn.Linear(120,256),
            nn.ReLU(),
            nn.Linear(256,8),
        )


    def forward(self, xyz, dirs):
        xyz = positional_encoding(xyz, 10, factor=2)
        dirs = positional_encoding(dirs, 2, factor=2)

        freq = self.hyper(xyz).unsqueeze(-1)
        
        x = self.l1(xyz)
        x = x * torch.sin(x * freq[:,0] + freq[:,1])

        x = self.l2(x)
        x = x * torch.sin(x * freq[:,2] + freq[:,3])


        x = self.l3(x)
        x = x * torch.sin(x * freq[:,4] + freq[:,5])

        x = torch.cat([x, dirs], dim=-1)

        x = self.l4(x)
        x = x * torch.sin(x * freq[:,6] + freq[:,7])

        x = self.l5(x)
        return x


class MLP_pc_opt(nn.Module):

    def __init__(self, dim):
        super(MLP_pc_opt, self).__init__()

        self.l1 = nn.Linear(60,256)
        self.l2 = nn.Linear(256,256) 
        self.l3 = nn.Linear(256,256) 
        self.l4 = nn.Linear(280,128)
        self.l5 = nn.Linear(128,dim)
        self.ac = nn.ReLU()

    def forward(self, xyz, dirs):
        xyz = positional_encoding(xyz, 10)
        dirs = positional_encoding(dirs, 4)

 
        x = self.l1(xyz)
        x = self.ac(x)

        x = self.l2(x)
        x = self.ac(x)

        x = self.l3(x)
        x = self.ac(x)

        x = torch.cat([x, dirs], dim=-1)

        x = self.l4(x)
        x = self.ac(x)

        x = self.l5(x)

        return x


class GatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding_mode='reflect', act_fun=nn.ReLU, normalization=nn.InstanceNorm2d):
        super().__init__()
        self.pad_mode = padding_mode
        self.filter_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        n_pad_pxl = int(self.dilation * (self.filter_size - 1) / 2)

        # this is for backward campatibility with older model checkpoints
        self.block = nn.ModuleDict(
            {
                'conv_f': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=n_pad_pxl),
                'act_f': act_fun(),
                'conv_m': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=n_pad_pxl),
                'act_m': nn.Sigmoid(),
                'norm': normalization(out_channels)
            }
        )

        # self.ac = nn.ReLU()

    def forward(self, x, *args, **kwargs):
        features = self.block.act_f(self.block.conv_f(x))
        mask = self.block.act_m(self.block.conv_m(x))
        output = features * mask
        output = self.block.norm(output)
        # output = self.ac(output)

        return output


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=GatedBlock):
        super().__init__()

        self.conv = conv_block(in_channels, out_channels)
        self.down = nn.AvgPool2d(2, 2)

    def forward(self, inputs, mask=None):
        outputs = self.down(inputs)
        outputs = self.conv(outputs, mask=mask)
        return outputs


class UpsampleBlock(nn.Module):
    def __init__(self, out_channels, upsample_mode, num_filt, conv_block=GatedBlock):
        super().__init__()

        #  = out_channels if same_num_filt else out_channels * 2
        if upsample_mode == 'deconv':
            self.up = nn.ConvTranspose2d(num_filt, out_channels, 4, stride=2, padding=1)
            self.conv = conv_block(out_channels * 2, out_channels)
        elif upsample_mode=='bilinear' or upsample_mode=='nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                        nn.Conv2d(num_filt, out_channels, 3, padding=1)
                                    )
            self.conv = conv_block(out_channels * 2, out_channels)
        else:
            assert False

    def forward(self, inputs1, inputs2):
        in1_up = self.up(inputs1)
        output= self.conv(torch.cat([in1_up, inputs2], 1))

        return output


class UNet(nn.Module):
    def __init__(self, args, out_dim=3, upsample_mode='nearest'):
        super().__init__()

        in_dim = args.dim - 3

        filters = [64, 128, 256, 512, 1024]
        filters = [x // 4 for x in filters]

        self.start = GatedBlock(in_dim, filters[0])

        self.down1 = DownsampleBlock(filters[0], filters[1])
        self.down2 = DownsampleBlock(filters[1], filters[2])
        self.down3 = DownsampleBlock(filters[2], filters[3])
        self.down4 = DownsampleBlock(filters[3], filters[4])

        self.up4 = UpsampleBlock(filters[3], upsample_mode, filters[4])
        self.up3 = UpsampleBlock(filters[2], upsample_mode, filters[3])
        self.up2 = UpsampleBlock(filters[1], upsample_mode, filters[2])
        self.up1 = UpsampleBlock(filters[0], upsample_mode, filters[1])

        self.final = nn.Sequential(
            nn.Conv2d(filters[0], out_dim, 1),
        )
        # self.U  = args.U

    def forward(self, x):

        in64 = self.start(x) 
        
        down1 = self.down1(in64) 
        down2 = self.down2(down1) 
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        up4 = self.up4(down4, down3)
        up3 = self.up3(up4, down2)
        up2 = self.up2(up3, down1)
        up1 = self.up1(up2, in64)
        
        return self.final(up1)

