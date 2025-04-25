import math
from mmcv.ops import ModulatedDeformConv2d
from ultralytics.utils.tal import dist2bbox, make_anchors
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Detect_dyhead']


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_swish(nn.Module):
    def __init__(self, inplace=False):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6


class DYReLU(nn.Module):
    def __init__(self, inp, oup, reduction=4, lambda_a=1.0, K2=True, use_bias=True, use_spatial=False,
                 init_a=[1.0, 0.0], init_b=[0.0, 0.0]):
        super(DYReLU, self).__init__()
        self.oup = oup
        self.lambda_a = lambda_a * 2
        self.K2 = K2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.use_bias = use_bias
        if K2:
            self.exp = 4 if use_bias else 2
        else:
            self.exp = 2 if use_bias else 1
        self.init_a = init_a
        self.init_b = init_b
        squeeze = inp // reduction if reduction == 4 else _make_divisible(inp // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(inp, squeeze),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze, oup * self.exp),
            h_sigmoid()
        )
        if use_spatial:
            self.spa = nn.Sequential(
                nn.Conv2d(inp, 1, kernel_size=1),
                nn.BatchNorm2d(1),
            )
        else:
            self.spa = None

    def forward(self, x):
        if isinstance(x, list):
            x_in = x[0]
            x_out = x[1]
        else:
            x_in = x
            x_out = x
        b, c, h, w = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup * self.exp, 1, 1)
        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
            a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
            b1 = b1 - 0.5 + self.init_b[0]
            b2 = b2 - 0.5 + self.init_b[1]
            out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
        elif self.exp == 2:
            if self.use_bias:
                a1, b1 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
                b1 = b1 - 0.5 + self.init_b[0]
                out = x_out * a1 + b1
            else:
                a1, a2 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
                a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
                out = torch.max(x_out * a1, x_out * a2)
        elif self.exp == 1:
            a1 = y
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
            out = x_out * a1

        if self.spa:
            ys = self.spa(x_in).view(b, -1)
            ys = F.softmax(ys, dim=1).view(b, 1, h, w) * h * w
            ys = F.hardtanh(ys, 0, 3, inplace=True) / 3
            out = out * ys

        return out


class Conv3x3Norm(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Conv3x3Norm, self).__init__()
        self.conv = ModulatedDeformConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.GroupNorm(num_groups=16, num_channels=out_channels)

    def forward(self, input, **kwargs):
        x = self.conv(input.contiguous(), **kwargs)
        x = self.bn(x)
        return x


class DyConv(nn.Module):
    def __init__(self, in_channels, out_channels, prev_channels=None, next_channels=None, conv_func=Conv3x3Norm):
        """
        prev_channels, next_channels: 邻域特征的通道数，如不匹配，则内部做 1x1 投影调整
        """
        super(DyConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.prev_channels = prev_channels if prev_channels is not None else in_channels
        self.next_channels = next_channels if next_channels is not None else in_channels

        self.DyConv = nn.ModuleList()
        # 顺序：0 用于后一层，1 当前层，2 前一层
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 2))
        self.AttnConv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.h_sigmoid = h_sigmoid()
        self.relu = DYReLU(in_channels, out_channels)
        # offset 用于当前层，要求输入通道为 in_channels
        self.offset = nn.Conv2d(in_channels, 27, kernel_size=3, stride=1, padding=1)
        # 如果前后邻域通道不等于当前，则添加投影层
        self.proj_prev = nn.Conv2d(self.prev_channels, in_channels, kernel_size=1) if self.prev_channels != in_channels else None
        self.proj_next = nn.Conv2d(self.next_channels, in_channels, kernel_size=1) if self.next_channels != in_channels else None

        self.init_weights()

    def init_weights(self):
        for m in self.DyConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.AttnConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, cur, prev=None, next_=None):
        # 若相邻特征通道不匹配，则先投影调整
        if prev is not None and prev.shape[1] != self.in_channels and self.proj_prev is not None:
            prev = self.proj_prev(prev)
        if next_ is not None and next_.shape[1] != self.in_channels and self.proj_next is not None:
            next_ = self.proj_next(next_)
        # 使用当前层特征计算 offset 与 mask
        offset_mask = self.offset(cur)
        offset = offset_mask[:, :18, :, :]
        mask = offset_mask[:, 18:, :, :].sigmoid()
        conv_args = dict(offset=offset, mask=mask)

        temp_fea = []
        # 当前层
        temp_fea.append(self.DyConv[1](cur, **conv_args))
        # 前一层
        if prev is not None:
            temp_fea.append(self.DyConv[2](prev, **conv_args))
        # 后一层
        if next_ is not None:
            t = self.DyConv[0](next_, **conv_args)
            t = F.interpolate(t, size=[cur.size(2), cur.size(3)])
            temp_fea.append(t)
        # 融合
        temp_fea = torch.stack(temp_fea)  # (n, B, C, H, W)
        attn_fea = torch.stack([self.AttnConv(fea) for fea in temp_fea])
        spa_pyr_attn = self.h_sigmoid(attn_fea)
        mean_fea = torch.mean(temp_fea * spa_pyr_attn, dim=0)
        out = self.relu(mean_fea)
        return out

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Detect_dyhead(nn.Module):
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=None):
        super().__init__()
        # 若未传入，则使用默认通道列表
        if ch is None or len(ch) == 0:
            ch = [256, 512, 1024]
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        # 为每个尺度构造一个 DyConv 模块，并以字符串形式存入 ModuleDict
        self.dyhead_tower = nn.ModuleDict()
        for i in range(self.nl):
            channel = ch[i]
            prev_ch = ch[i - 1] if i > 0 else channel
            next_ch = ch[i + 1] if i < self.nl - 1 else channel
            self.dyhead_tower[str(i)] = DyConv(channel, channel,
                                               prev_channels=prev_ch,
                                               next_channels=next_ch,
                                               conv_func=Conv3x3Norm)

    def forward(self, x):
        # x 为列表，每个元素为不同尺度特征
        new_feats = []
        for i, feat in enumerate(x):
            prev_feat = x[i - 1] if i > 0 else None
            next_feat = x[i + 1] if i < self.nl - 1 else None
            new_feat = self.dyhead_tower[str(i)](feat, prev=prev_feat, next_=next_feat)
            new_feats.append(new_feat)
        x = new_feats  # list of updated features
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export:
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)


if __name__ == "__main__":
    feat1 = torch.rand(1, 256, 80, 80)
    feat2 = torch.rand(1, 512, 40, 40)
    feat3 = torch.rand(1, 1024, 20, 20)
    model = Detect_dyhead()
    model.train()  # 测试训练模式下的输出
    out = model([feat1, feat2, feat3])
    if isinstance(out, tuple):
        y, features = out
        print("y shape:", y.shape)
        for i, f in enumerate(features):
            print(f"feature {i} shape:", f.shape)
    elif isinstance(out, list):
        for i, tensor in enumerate(out):
            print(f"Output feature {i} shape:", tensor.shape)
    else:
        print("Output shape:", out.shape)
