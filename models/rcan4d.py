# RCAN-4D for turbulence SR
# - Original RCAN was Conv3d on [B, C, D, H, W]
# - Now you want Conv4d on        [B, 1, C, H, W, D]  (B 1 C H W D)
#   We interpret it as Conv4d input: [B, in_ch=1, L=C, D=H, H=W, W=D]
#
# Notes:
# 1) Upsampling only on the last 3 dims (D,H,W) i.e. (H,W,D in your notation). L (=C) is NOT upscaled.
# 2) Channel-attention pools over (L,D,H,W).
#
# You can drop this file as a standalone module.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple
class Conv4d(nn.Module):
    """
    Input : [B, Cin, L, D, H, W]
    Output: [B, Cout, L', D', H', W']
    Implemented as a stack of Conv3d along the L-kernel axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=(1, 1, 1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1, 1, 1),
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if padding_mode not in {"zeros"}:
            raise ValueError(f"padding_mode must be 'zeros', got {padding_mode}")
        assert len(kernel_size) == 4
        assert len(stride) == 4
        assert len(padding) == 4
        assert len(dilation) == 4
        assert groups == 1, "groups != 1 not implemented"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        # master weight for init; then sliced into conv3d layers
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        self.reset_parameters()

        self.conv3d_layers = nn.ModuleList()
        for i in range(self.kernel_size[0]):
            conv3d_layer = nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size[1:],
                padding=self.padding[1:],
                dilation=self.dilation[1:],
                stride=self.stride[1:],
                bias=False,
            )
            # share (slice) weights
            conv3d_layer.weight = nn.Parameter(self.weight[:, :, i, :, :, :])
            self.conv3d_layers.append(conv3d_layer)

        # delete the master tensor to avoid double params
        del self.weight

    def reset_parameters(self) -> None:
        # init the master weight (before del) if exists
        if hasattr(self, "weight"):
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
        else:
            # conv3d layers will be initialized by their own reset if needed
            pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: [B, Cin, L, D, H, W]
        (B, Cin, l_i, d_i, h_i, w_i) = tuple(input.shape)
        (l_k, d_k, h_k, w_k) = self.kernel_size
        (l_p, d_p, h_p, w_p) = self.padding
        (l_d, d_d, h_d, w_d) = self.dilation
        (l_s, d_s, h_s, w_s) = self.stride

        l_o = (l_i + 2 * l_p - l_k - (l_k - 1) * (l_d - 1)) // l_s + 1
        d_o = (d_i + 2 * d_p - d_k - (d_k - 1) * (d_d - 1)) // d_s + 1
        h_o = (h_i + 2 * h_p - h_k - (h_k - 1) * (h_d - 1)) // h_s + 1
        w_o = (w_i + 2 * w_p - w_k - (w_k - 1) * (w_d - 1)) // w_s + 1

        out = input.new_zeros((B, self.out_channels, l_o, d_o, h_o, w_o))

        for i in range(l_k):
            zero_offset = -l_p + (i * l_d)
            j_start = max(zero_offset % l_s, zero_offset)
            j_end = min(l_i, l_i + l_p - (l_k - i - 1) * l_d)
            for j in range(j_start, j_end, l_s):
                out_frame = (j - zero_offset) // l_s
                out[:, :, out_frame, :, :, :] += self.conv3d_layers[i](input[:, :, j, :, :, :])

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1, 1)
        return out

def initialize_weights(m: nn.Module, activation: str = "relu", a=None, scale: float = 1.0):
    """
    Mimic old initialize_weights:
    - For Conv4d: init each internal Conv3d weight, then scale.
    - For Conv3d / ConvTranspose3d / Linear: kaiming_normal_ then scale.
    """
    if isinstance(m, Conv4d):
        for conv3d in m.conv3d_layers:    
            nn.init.kaiming_normal_(conv3d.weight, nonlinearity=activation, a=a)
            conv3d.weight.data *= scale
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
        return

    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity=activation, a=a)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
        m.weight.data *= scale
        return
    return

class PixelShuffle3d(nn.Module):
    """
    Input : [B, C, L, D, H, W]
    Upscale D/H/W by r (NOT L).
    Requires C to be divisible by r^3.
    """

    def __init__(self, scale: int):
        super().__init__()
        self.scale = int(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L, D, H, W = x.shape
        r = self.scale
        assert C % (r ** 3) == 0, f"channels {C} must be divisible by r^3={r**3}"
        Cout = C // (r ** 3)

        x = x.contiguous().view(B, Cout, r, r, r, L, D, H, W)
        # [B, Cout, rD, rH, rW, L, D, H, W] -> [B, Cout, L, D*r, H*r, W*r]
        x = x.permute(0, 1, 5, 6, 2, 7, 3, 8, 4).contiguous()
        return x.view(B, Cout, L, D * r, H * r, W * r)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    k = _quadruple(kernel_size)
    p = tuple(kk // 2 for kk in k)
    return Conv4d(in_channels, out_channels, kernel_size=k, padding=p, bias=bias)

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

# -------------------------
# Channel Attention (4D)
# -------------------------
class CALayer4d(nn.Module):
    def __init__(self, channel, reduction=16, conv4d=default_conv):
        super().__init__()
        self.reduction = reduction
        c1 = max(1, channel // reduction)
        conv1 = conv4d(channel, c1, 1, bias=True)
        conv2 = conv4d(c1, channel, 1, bias=True)
        initialize_weights(conv1, "relu", None, 0.1)
        initialize_weights(conv2, "linear", None, 0.1)

        self.conv_du = nn.Sequential(
            conv1,
            nn.ReLU(inplace=True),
            conv2,
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L, D, H, W] -> global average pool over (L,D,H,W)
        y = x.mean(dim=(2, 3, 4, 5), keepdim=True)
        y = self.conv_du(y)
        return x * y


class RCAB4d(nn.Module):
    def __init__(self, conv4d, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super().__init__()
        modules_body = []
        for i in range(2):
            c = conv4d(n_feat, n_feat, kernel_size, bias=bias)
            initialize_weights(c, "relu" if i == 0 else "linear", None, 0.1)
            modules_body.append(c)
            if bn:
                modules_body.append(nn.BatchNorm3d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer4d(n_feat, reduction, conv4d=conv4d))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return res + x


class ResidualGroup4d(nn.Module):
    def __init__(self, conv4d, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super().__init__()
        modules_body = [
            RCAB4d(conv4d, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
            for _ in range(n_resblocks)
        ]
        c = conv4d(n_feat, n_feat, kernel_size, bias=True)
        initialize_weights(c, "linear", None, 0.1)
        modules_body.append(c)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return res + x


class RCAN4D(nn.Module):
    """
    Input : [B, 1, C, H, W, D]
    Internally we use: [B, n_feats, L=C, D=H, H=W, W=D]
    Output: [B, 1, C, H*up, W*up, D*up]  (same layout as input, with spatial SR)
    """

    def __init__(self, n_resgroups, n_resblocks, n_feats, reduction, scale, n_colors=1, res_scale=1, conv4d=default_conv):
        super().__init__()
        self.upscale = scale
        kernel_size = 3
        act = nn.ReLU(True)

        # head
        conv1 = conv4d(n_colors, n_feats, kernel_size, bias=True)
        initialize_weights(conv1, "linear", None, 0.1)
        self.head = nn.Sequential(conv1)

        # body
        body = [
            ResidualGroup4d(conv4d, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks)
            for _ in range(n_resgroups)
        ]
        conv2 = conv4d(n_feats, n_feats, kernel_size, bias=True)
        initialize_weights(conv2, "linear", None, 0.1)
        body.append(conv2)
        self.body = nn.Sequential(*body)

        # tail (upsample only last-3 dims)
        conv3 = conv4d(n_feats, n_colors, kernel_size, bias=True)
        initialize_weights(conv3, "linear", None, 0.1)
        self.tail = nn.Sequential(
            Upsampler(conv4d, scale, n_feats, act=False),
            conv3,
        )
        self.conv0 = nn.Conv3d(4, 8, kernel_size=1, padding=0)
        initialize_weights(self.conv0, 'linear', None, 0.1)
        self.L_reduce = nn.Conv3d(
            in_channels=8,
            out_channels=4,
            kernel_size=1,
            padding=0
        )
        initialize_weights(self.L_reduce, "linear", None, 0.1)

    @staticmethod
    def _to_internal(x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, C, H, W, D] -> [B, 1, L=C, D=H, H=W, W=D]
        # matches Conv4d ordering [B, Cin, L, D, H, W]
        return x.unsqueeze(1)  # -> [B,1,C,H,W,D]
        # return x

    @staticmethod
    def _to_external(x: torch.Tensor) -> torch.Tensor:
        # internal already matches external layout in this design
        # return x
        return x.squeeze(1)

    def forward(self, x):
        # print(x.shape)
        x = self.conv0(x)
        x = self._to_internal(x)
        x = self.head(x)
        res = self.body(x)
        res = res + x
        B, C, L, D, H, W = res.shape

        # 把 L 当 channel
        res = res.permute(0, 1, 3, 4, 5, 2).contiguous()   # [B, C, D, H, W, L]
        res = res.view(B * C, D, H, W, L).permute(0, 4, 1, 2, 3)  # [B*C, L, D, H, W]

        # 用 Conv3d(8→4)
        res = self.L_reduce(res)        # [B*C, 4, D, H, W]

        # 还原回 4D conv layout
        res = res.permute(0, 2, 3, 4, 1).contiguous()     # [B*C, D, H, W, 4]
        res = res.view(B, C, D, H, W, 4).permute(0, 1, 5, 2, 3, 4)   # [B, C, 4, D, H, W]
        out = self.tail(res)
        out = self._to_external(out)
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if "tail" in name:
                        print("Replace pre-trained upsampler to new one...")
                    else:
                        raise RuntimeError(
                            f"While copying parameter {name}, model has {own_state[name].size()} but ckpt has {param.size()}"
                        )
            elif strict:
                if "tail" not in name:
                    raise KeyError(f'unexpected key "{name}" in state_dict')

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError(f'missing keys in state_dict: "{missing}"')


def init_rcan4d(approx_param: str, upscale: int):
    """
    Keep your old approx_param presets, but for 4D network.
    Input/Output channels (n_colors) is 1 because your tensor is [B,1,C,H,W,D].
    """
    if approx_param == "0.5M":
        n_feats, n_resblocks, n_resgroups = 26, 1, 1
    elif approx_param == "0.8M":
        n_feats, n_resblocks, n_resgroups = 20, 1, 1
    # elif approx_param == "1.4M":
    #     n_feats, n_resblocks, n_resgroups = 44, 1, 1
    elif approx_param == "17M":
        n_feats, n_resblocks, n_resgroups = 37, 20, 3
    # elif approx_param == "2.7M":
    #     n_feats, n_resblocks, n_resgroups = 60, 1, 1
    # elif approx_param == "5M":
    #     n_feats, n_resblocks, n_resgroups = 64, 10, 1
    # elif approx_param == "11M":
    #     n_feats, n_resblocks, n_resgroups = 64, 20, 2
    # elif approx_param == "17M":
    #     n_feats, n_resblocks, n_resgroups = 64, 20, 3
    # elif approx_param == "50M":
    #     n_feats, n_resblocks, n_resgroups = 64, 20, 10
    else:
        raise ValueError("Please provide the correct approx_param")

    return RCAN4D(
        n_feats=n_feats,
        n_colors=1,
        n_resgroups=n_resgroups,
        n_resblocks=n_resblocks,
        reduction=16,
        scale=upscale,
        res_scale=1,
    )


# -------------------------
# quick sanity check
# -------------------------
if __name__ == "__main__":
    # Example: C=4 channels moved into L-dim, and spatial is (H,W,D)=(16,16,16)
    x = torch.randn(2, 4, 16, 16, 16).cuda()
    net = init_rcan4d("17M", upscale=8).cuda()
    y = net(x)
    print("x:", x.shape, "y:", y.shape)  # expect [2,1,4,32,32,32]
