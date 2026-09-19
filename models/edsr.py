import math
import torch
from torch import nn

def initialize_weights(m,activation,a,scale):
    if isinstance(m, torch.nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity=activation,a=a)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.ConvTranspose3d):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity=activation,a=a)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity=activation,a=a)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    else:
        print('no init for ',m)
        pass
    m.weight.data *= scale

class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 8 * n_feat, 3, bias))
                initialize_weights(m[-1], 'relu', None, 0.1)
                m.append(PixelShuffle3d(2))
                if bn: m.append(nn.BatchNorm3d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 27 * n_feat, 3, bias))
            initialize_weights(m[-1], 'relu', None, 0.1)
            m.append(PixelShuffle3d(3))
            if bn: m.append(nn.BatchNorm3d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if i ==0: 
                initialize_weights(m[-1], 'relu', None, 0.1)
            else:
                initialize_weights(m[-1], 'linear', None, 0.1)
            if bn:
                m.append(nn.BatchNorm3d(n_feats))
            if i == 0:
                m.append(act)
            

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class EDSR(nn.Module):
    def __init__(self, in_channels,n_feats,kernel_size,res_scale,n_resblocks,scale, conv=default_conv):
        super(EDSR, self).__init__()

        
        # define head module
        m_head = [conv(in_channels, n_feats, kernel_size)]
        initialize_weights(m_head[-1], 'linear', None, 0.1)

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, 
                act=nn.ReLU(True), res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        initialize_weights(m_body[-1], 'linear', None, 0.1)
        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, in_channels, kernel_size)
        ]
        initialize_weights(m_tail[-1], 'linear', None, 0.1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.upscale = scale

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x 
    
def init_edsr(approx_param,upscale):
    if approx_param == '0.5M':
        n_feats = 14
    elif approx_param == '0.8M':
        n_feats = 20
    elif approx_param == '1.4M':
        n_feats = 24
    elif approx_param == '2.7M':
        n_feats = 34
    elif approx_param == '5M':
        n_feats = 46
    elif approx_param == '11M':
        n_feats = 68
    elif approx_param == '17M':
        n_feats = 86
    elif approx_param == '35M':
        n_feats = 120
    return EDSR(in_channels=4,n_feats=n_feats,kernel_size=3,res_scale=0.1,n_resblocks=32,scale=upscale)
