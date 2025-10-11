import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from pytorch_wavelets import DWTForward, DWTInverse
from einops import rearrange
from dropblock import LinearScheduler, DropBlock2D

class Conv1Relu(nn.Module):  # 1*1卷积用来降维
    def __init__(self, in_ch, out_ch):
        super(Conv1Relu, self).__init__()
        self.extract = nn.Sequential(nn.Conv2d(in_ch, out_ch, (1, 1), bias=False),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.extract(x)
        return x
    
class ChannelChecker(nn.Module):
    def __init__(self, backbone, inplanes, input_size):
        super(ChannelChecker, self).__init__()
        input_sample = torch.randn(1, 3, input_size, input_size)
        f1, f2, f3, f4 = backbone(input_sample)

        channels1 = f1.size(1)
        channels2 = f2.size(1)
        channels3 = f3.size(1)
        channels4 = f4.size(1)

        self.conv1 = Conv1Relu(channels1, inplanes) if (channels1 != inplanes) else None
        self.conv2 = Conv1Relu(channels2, inplanes*2) if (channels2 != inplanes*2) else None
        self.conv3 = Conv1Relu(channels3, inplanes*4) if (channels3 != inplanes*4) else None
        self.conv4 = Conv1Relu(channels4, inplanes*8) if (channels4 != inplanes*8) else None

        if (channels1 != inplanes) or (channels2 != inplanes*2) or \
                (channels3 != inplanes*4) or (channels4 != inplanes*8):
            print("\n*** Please note that the channel of features from backbone was automatically modified ***")
            print("*** {} -->--> {} ***".format([channels1, channels2, channels3, channels4],
                                                [inplanes*1, inplanes*2, inplanes*4, inplanes*8]))

    def forward(self, f1, f2, f3, f4):
        f1 = self.conv1(f1) if (self.conv1 is not None) else f1
        f2 = self.conv2(f2) if (self.conv2 is not None) else f2
        f3 = self.conv3(f3) if (self.conv3 is not None) else f3
        f4 = self.conv4(f4) if (self.conv4 is not None) else f4

        return f1, f2, f3, f4


class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, bn_momentum=0.0003):
        super().__init__()
        inter_channels = in_channels // 4


        self.last_conv = nn.Sequential(nn.Conv2d(in_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(inter_channels, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(inter_channels, momentum=bn_momentum),
                                       nn.ReLU(),
                                       )


        self.classify = nn.Conv2d(in_channels=inter_channels, out_channels= out_channels, kernel_size=1,
                                        stride=1, padding=0, dilation=1, bias=True)

    def forward(self, x):
       
        x = self.last_conv(x)
        pred = self.classify(x)
        return pred
    
class DropBlock(nn.Module):
    """
    [Ghiasi et al., 2018] DropBlock: A regularization method for convolutional networks
    """
    def __init__(self, rate=0.15, size=7, step=50):
        super().__init__()

        self.drop = LinearScheduler(
            DropBlock2D(block_size=size, drop_prob=0.),
            start_value=0,
            stop_value=rate,
            nr_steps=step
        )

    def forward(self, feats: list):
        if self.training:  # 只在训练的时候加上dropblock
            for i, feat in enumerate(feats):
                feat = self.drop(feat)
                feats[i] = feat
        return feats

    def step(self):
        self.drop.step()


class Conv3Relu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Conv3Relu, self).__init__()
        self.extract = nn.Sequential(nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1),
                                               stride=(stride, stride), bias=False),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.extract(x)
        return x
    
class DropBlock(nn.Module):
    """
    [Ghiasi et al., 2018] DropBlock: A regularization method for convolutional networks
    """
    def __init__(self, rate=0.15, size=7, step=50):
        super().__init__()

        self.drop = LinearScheduler(
            DropBlock2D(block_size=size, drop_prob=0.),
            start_value=0,
            stop_value=rate,
            nr_steps=step
        )
        # print('-' * 100)
        # print('dropblock is initialized successfully!')
        # print('block_size={}, drop_prob={}, step={}'.format(size, rate, step))

    def forward(self, feats: list):
        if self.training:  # 只在训练的时候加上dropblock
            for i, feat in enumerate(feats):
                feat = self.drop(feat)
                feats[i] = feat
        return feats

    def step(self):
        self.drop.step()
        # print("drop_prob = {}".format(self.drop.dropblock.drop_prob))

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=(6, 12, 18)):
        super(ASPP, self).__init__()

        rate1, rate2, rate3 = tuple(atrous_rates)

        out_channels = int(in_channels / 2)

        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=rate1, dilation=rate1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=rate2, dilation=rate2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=rate3, dilation=rate3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))

        # 全局平均池化
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

        self.dim_reduction = Conv3Relu(out_channels * 5, in_channels)

    def forward(self, x):
        h, w = x.shape[-2:]

        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)

        feat4 = F.interpolate(self.gap(x), (h, w), mode="bilinear", align_corners=True)

        out = self.dim_reduction(torch.cat((feat0, feat1, feat2, feat3, feat4), 1))

        return out

class AlignedModulev2PoolingAtten(nn.Module):

    def __init__(self, inplanel,inplaneh, outplane, kernel_size=3):
        super(AlignedModulev2PoolingAtten, self).__init__()
        self.down_h = nn.Conv2d(inplaneh, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplanel, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 4, kernel_size=kernel_size, padding=1, bias=False)
        self.flow_gate = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=kernel_size, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1,x2):
        low_feature=x1
        h_feature = x2
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        l_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)

        flow = self.flow_make(torch.cat([h_feature, l_feature], 1))
        flow_up, flow_down = flow[:, :2, :, :], flow[:, 2:, :, :]

        h_feature_warp = self.flow_warp(h_feature_orign, flow_up, size=size)
        l_feature_warp = self.flow_warp(low_feature, flow_down, size=size)

        h_feature_mean = torch.mean(h_feature, dim=1).unsqueeze(1)
        l_feature_mean = torch.mean(low_feature, dim=1).unsqueeze(1)
        h_feature_max = torch.max(h_feature, dim=1)[0].unsqueeze(1)
        l_feature_max = torch.max(low_feature, dim=1)[0].unsqueeze(1)

        flow_gates = self.flow_gate(torch.cat([h_feature_mean, l_feature_mean, h_feature_max, l_feature_max], 1))

        fuse_feature = h_feature_warp * flow_gates + l_feature_warp * (1 - flow_gates)

        return fuse_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            normal_layer(out_planes),
            nn.ReLU(inplace=True),
    )

class ModReLU(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.b = nn.Parameter(torch.Tensor(features))
        self.b.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        return torch.abs(x) * F.relu(torch.cos(torch.angle(x) + self.b))

class Frequency_Modeling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.filter = nn.Linear(dim, dim)
        self.modrelu = ModReLU(dim)

    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        B,C,H,W = x.shape
        x = x.view(B,C,H*W).permute(0,2,1)
        x_fft = torch.fft.fft(x, dim=1)  # FFT along the sequence dimension
        x_filtered = self.filter(x_fft.real) + 1j * self.filter(x_fft.imag)
        x_filtered = self.modrelu(x_filtered)
        x_out = torch.fft.ifft(x_filtered, dim=1).real
        x_out = x_out.permute(0,2,1).view(B,C,H,W)
        return x_out

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Orthogonal_Directional_Priors(nn.Module):  
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''
    
    def __init__(self, c1, c2, k, s):
        super().__init__()

        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ODFM(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.conv1_1 = BasicConv2d(embed_dim * 2, embed_dim, 1)
        self.local_11conv = nn.Conv2d(input_dim // 2, embed_dim, 1)
        self.global_11conv = nn.Conv2d(input_dim // 2, embed_dim, 1)
        self.odp = Orthogonal_Directional_Priors(embed_dim,embed_dim,k=3, s=1)
        self.fm = Frequency_Modeling(embed_dim)
        
    def forward(self, x):
        x_0, x_1 = x.chunk(2, dim=1) # x_0 torch.Size([1, 512, 8, 8])
        x_0 = self.fm(self.local_11conv(x_0)) #local: torch.Size([1, 256, 8, 8])
        x_1 = self.odp(self.global_11conv(x_1)) #ghpa
        x = torch.cat([x_0, x_1], dim=1)
        x = self.conv1_1(x)

        return x

class ODFM_fusion(nn.Module):
    def __init__(self,dim):
        super(ODFM_fusion,self).__init__()
        self.attention = ODFM(dim*4,dim)
    def forward(self,feature_list):
        for i,feature in enumerate(feature_list[:-1]):
            feature = F.pixel_shuffle(feature,1)
            x = feature if i ==0 else torch.cat([x,feature],dim=1)
        x = torch.cat([x,feature_list[-1]],dim=1)
        x = self.attention(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.stage1_Conv1 = Conv3Relu(inplanes * 1, inplanes)  # channel: 2*inplanes ---> inplanes
        self.stage2_Conv1 = Conv3Relu(inplanes * 2, inplanes * 2)  # channel: 4*inplanes ---> 2*inplanes
        self.stage3_Conv1 = Conv3Relu(inplanes * 4, inplanes * 4)  # channel: 8*inplanes ---> 4*inplanes
        self.stage4_Conv1 = Conv3Relu(inplanes * 8, inplanes * 8)  # channel: 16*inplanes ---> 8*inplanes

        self.stage2_Conv_after_up = Conv3Relu(inplanes * 2, inplanes)
        self.stage3_Conv_after_up = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage4_Conv_after_up = Conv3Relu(inplanes * 8, inplanes * 4)
        
        self.stage1_Conv2 = Conv3Relu(inplanes * 2, inplanes)
        self.stage2_Conv2 = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage3_Conv2 = Conv3Relu(inplanes * 8, inplanes * 4)
        
        self.scn41= AlignedModulev2PoolingAtten(inplanes , inplanes, inplanes)
        self.scn31= AlignedModulev2PoolingAtten(inplanes , inplanes, inplanes)
        self.scn21= AlignedModulev2PoolingAtten(inplanes , inplanes, inplanes)
        self.final_Conv5 = Conv3Relu(inplanes , inplanes)
        
        self.fusion_module = ODFM_fusion(inplanes)

        self.expand_field = ASPP(inplanes * 8)

        self.stage2_Conv3 = Conv3Relu(inplanes * 2, inplanes)   # 降维
        self.stage3_Conv3 = Conv3Relu(inplanes * 4, inplanes)
        self.stage4_Conv3 = Conv3Relu(inplanes * 8, inplanes)
        self.final_Conv = Conv3Relu(inplanes * 4, inplanes)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        rate, size, step = (0.15, 7, 30)
        self.drop = DropBlock(rate=rate, size=size, step=step)

    def forward(self, ms_feats):
        fa1, fa2, fa3, fa4 = ms_feats
        feature1_h, feature1_w = fa1.size(2), fa1.size(3)

        [fa1, fa2, fa3, fa4] = self.drop([fa1, fa2, fa3, fa4])  # dropblock

        feature1 = self.stage1_Conv1(torch.cat([fa1], 1))  # inplanes
        feature2 = self.stage2_Conv1(torch.cat([fa2], 1))  # inplanes * 2
        feature3 = self.stage3_Conv1(torch.cat([fa3], 1))  # inplanes * 4
        feature4 = self.stage4_Conv1(torch.cat([fa4], 1))  # inplanes * 8
        if self.expand_field is not None:
            feature4 = self.expand_field(feature4)

        feature3_2 = self.stage4_Conv_after_up(self.up(feature4))
        feature3 = self.stage3_Conv2(torch.cat([feature3, feature3_2], 1)) #torch.Size([4, 512, 16, 16])

        feature2_2 = self.stage3_Conv_after_up(self.up(feature3))
        feature2 = self.stage2_Conv2(torch.cat([feature2, feature2_2], 1)) #torch.Size([4, 256, 32, 32])

        feature1_2 = self.stage2_Conv_after_up(self.up(feature2))
        feature1 = self.stage1_Conv2(torch.cat([feature1, feature1_2], 1)) #torch.Size([4, 128, 64, 64])

        feature4=self.scn41(feature1, self.stage4_Conv3(feature4))
        feature3=self.scn31(feature1, self.stage3_Conv3(feature3))
        feature2=self.scn21(feature1, self.stage2_Conv3(feature2))
        feature = self.fusion_module([feature1,feature2,feature3,feature4])

        return feature
    
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SAConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
                                        nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1,
                                                  groups=in_channel, bias=False))
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = Conv(in_channel, out_channel, k=kernel_size, s=kernel_size, p=0)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)


class SWAStage(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')  # 小波下采样
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.sa = SAConv(in_channel=out_ch, out_channel=out_ch, kernel_size=3)

    def forward(self, x):
        yL, yH = self.dwt(x)
        y_HL = yH[0][:, :, 0, :, :]
        y_LH = yH[0][:, :, 1, :, :]
        y_HH = yH[0][:, :, 2, :, :]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)  # 通道数 ×4
        x = self.sa(self.conv(x))
        return x

class SWABackbone(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.stage1 = SWAStage(3, base_channels)           # RGB输入
        self.stage2 = SWAStage(base_channels, base_channels * 2)
        self.stage3 = SWAStage(base_channels * 2, base_channels * 4)
        self.stage4 = SWAStage(base_channels * 4, base_channels * 8)

    def forward(self, x):
        f1 = self.stage1(x)  # (B, C, H/2, W/2)
        f2 = self.stage2(f1) # (B, 2C, H/4, W/4)
        f3 = self.stage3(f2) # (B, 4C, H/8, W/8)
        f4 = self.stage4(f3) # (B, 8C, H/16, W/16)
        return [f1, f2, f3, f4]
    
class ISWAStage(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.idwt = DWTInverse(mode='zero', wave='haar')
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 4, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x 的通道数 = cL + cH*3
        b, c, h, w = x.shape
        c_per = c // 4  # 每个子带通道数

        yL  = x[:, :c_per, :, :]
        yHL = x[:, c_per:2*c_per, :, :]
        yLH = x[:, 2*c_per:3*c_per, :, :]
        yHH = x[:, 3*c_per:4*c_per, :, :]

        # 组回 pytorch_wavelets 需要的形式
        yH = [(torch.stack([yHL, yLH, yHH], dim=2))]  
        out = self.idwt((yL, yH))   # 逆变换
        out = self.conv(out)
        return out
                
class Seg_Detection(nn.Module):
    def __init__(self, dim=64,
                input_size=256, pretrain=''):
        super().__init__()
        self.inplanes = 64

        self._create_backbone()
        self._create_neck()
        self._create_heads()

        self.check_channels = ChannelChecker(self.backbone, self.inplanes, input_size)

        if pretrain.endswith(".pt"):
            self._init_weight(pretrain)   # todo:这里预训练初始化和 hrnet主干网络的初始化有冲突，必须要改！

    def forward(self, x):
        _, _, h_input, w_input = x.shape
       
        f1, f2, f3, f4 = self.backbone(x)  # feature_a_1: 输入图像a的最大输出特征图
        f1, f2, f3, f4 = self.check_channels(f1, f2, f3, f4)  # 检查通道数是否一致
        ms_feats = f1, f2, f3, f4  # 多尺度特征

        feature = self.neck(ms_feats)

        out = self.head_forward(feature , out_size=(h_input, w_input))

        return out
    
    def forward_att_hook(self, x):
            _, _, h_input, w_input = x.shape
        
            f1, f2, f3, f4 = self.backbone(x)  # feature_a_1: 输入图像a的最大输出特征图
            
            block1 = self.block(self.inplanes)
            block2 = self.block(self.inplanes * 2)
            block3 = self.block(self.inplanes * 2**2)
            block4 = self.block(self.inplanes * 2**3)

            ms_feats = block1(f1), block2(f2), block3(f3), block4(f4)  # 多尺度特征 

            feature = self.neck(ms_feats)

            out = self.head_forward(feature , out_size=(h_input, w_input))

            return out, ms_feats
   

    def head_forward(self, feature , out_size):  

        out = F.interpolate(self.head(feature ), size=out_size, mode='bilinear', align_corners=True)

        return out
  

    def _init_weight(self, pretrain=''):  # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 只要是卷积都操作，都对weight和bias进行kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):  # bn层都权重初始化为1， bias=0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if pretrain.endswith('.pt'):
            pretrained_dict = torch.load(pretrain)
            if isinstance(pretrained_dict, nn.DataParallel):
                pretrained_dict = pretrained_dict.module
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.state_dict().items()
                                if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(OrderedDict(model_dict), strict=True)
            print("=> ChangeDetection load {}/{} items from: {}".format(len(pretrained_dict),
                                                                        len(model_dict), pretrain))


    def _create_backbone(self):
        self.backbone = SWABackbone(base_channels = self.inplanes)   
        

    def _create_neck(self):
        self.neck = Decoder(self.inplanes)

    def _create_heads(self):
        self.head = FCNHead(self.inplanes, 2)

from thop import profile, clever_format
if __name__ == "__main__":
    device = torch.device('cuda:0')
    model = Seg_Detection(dim=64,
                          ).to(device)

    # 随机输入
    input = torch.randn(4, 3, 256, 256).to(device)
    output = model(input)

    # 统计 FLOPs 和 Params（用单张图片来算更直观）
    dummy = torch.randn(1, 3, 256, 256).to(device)
    flops, params = profile(model, inputs=(dummy, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print(model)
    print(f"Output shape: {output.shape}")
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
