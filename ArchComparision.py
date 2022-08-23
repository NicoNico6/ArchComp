import argparse
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchtoolbox.tools import summary

__all__ = ['birealnet', 'reactnet', 'boolnetv1', 'boolnetv2']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def OR(x, y):  # -1,1
    """Logic OR"""
    y = y.add(1).div(2)  # 0,1
    x = x.add(1).div(2)  # 0,1
    return x.add(y).clamp(0, 1).mul(2).add(-1)


def XNOR(x, y):  # -1,1
    """Logic XNOR"""
    y = x.mul(y)
    return y


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_chn, 1, 1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class GhostSign(nn.Module):
    def __init__(self, channels, slices=4, mode="uniform"):
        super(GhostSign, self).__init__()
        self.channels = channels
        assert (slices == 1) or (slices > 0 and (slices % 2 == 0)), "the number of slics must be even or one"

        self.k = slices // 2

        self.mode = mode

        self.register_buffer("temperature", torch.Tensor([1]))

        slice_1 = []
        slice_2 = []

        for i in range(-self.k, self.k + 1):
            if i != 0 and self.k != 0:
                if i < 0:
                    index = i + self.k
                    if self.mode == "uniform":
                        slice_1.append(1.0 / float((self.k + 1)) * i)

                    elif self.mode == "non_uniform":
                        slice_negtive = -(2 ** (-(2 ** int(math.log(self.k) / math.log(2))) + abs(i) - 1))
                        slice_1.append(slice_negtive)

                    else:
                        raise ValueError

                elif i > 0:
                    index = i + self.k - 1
                    if self.mode == "uniform":
                        slice_2.append(1.0 / float((self.k + 1)) * i)

                    elif self.mode == "non_uniform":
                        slice_positive = (2 ** (-(2 ** (int(math.log(self.k) / math.log(2)))) + abs(i) - 1))
                        slice_2.append(slice_positive)

                    else:
                        raise ValueError

        if len(slice_1 + slice_2) != 0:
            self.slice_1 = (torch.Tensor(slice_1).reshape(1, -1, 1, 1, 1))
            self.slice_2 = (torch.Tensor(slice_2).reshape(1, -1, 1, 1, 1))
        else:
            self.slice_1 = torch.zeros(1, 1, 1, 1, 1)
            self.slice_2 = torch.zeros(1, 1, 1, 1, 1)

        self.binarize = BinaryActivation(ste="Hardtanh")

    def update_temperature(self):
        self.temperature *= 0.965

    def forward(self, x):

        assert len(x.size()) == 4, "only support 4-D(N C H W) inputs"

        if not self.k == 0:
            slice = torch.cat(((self.slice_1.to(device="cuda" if x.is_cuda else "cpu")),
                               (self.slice_2.to(device="cuda" if x.is_cuda else "cpu"))), dim=1)

        else:
            slice = self.slice_1.to(device="cuda" if x.is_cuda else "cpu")

        x = x.unsqueeze(1) + slice

        x = self.binarize(x)

        return x


class BinaryActivation(nn.Module):
    def __init__(self, ste="Hardtanh"):
        super(BinaryActivation, self).__init__()
        self.ste = ste
        assert self.ste in {"Hardtanh", "Polynomial"}

    def polynomial_forward(self, x):
        out_forward = torch.sign(x)
        if not self.training:
            return out_forward

        # out_e1 = (x^2 + 2*x)
        # out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x * x + 2 * x) * (1 - mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x * x + 2 * x) * (1 - mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

    def hardtanh_forward(self, x):
        out_forward = torch.sign(x)
        if not self.training:
            return out_forward

        out = x.clamp(-1, 1)

        out = out_forward.detach() - out.detach() + out

        return out

    def forward(self, x):
        if self.ste == "Hardtanh":
            return self.hardtanh_forward(x)

        elif self.ste == "Polynomial":
            return self.polynomial_forward(x)


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_chn, 1, 1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, groups=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.number_of_weights = in_chn // groups * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn // groups, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights), dim=3, keepdim=True), dim=2, keepdim=True),
                                    dim=1, keepdim=True)
        scaling_factor = scaling_factor.detach()

        if not self.training:
            binary_weights_no_grad = scaling_factor * torch.sign(real_weights)

            return F.conv2d(x, binary_weights_no_grad, stride=self.stride, padding=self.padding, groups=self.groups)

        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)

        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)

        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights

        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding, groups=self.groups)

        return y


class MultiBConv(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=64,
                 kernel_size=2,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 wb=True,
                 ):
        super(MultiBConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = dilation
        self.dilation = dilation
        self.groups = groups
        self.wb = wb

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        self.binarize = BinaryActivation(ste="Hardtanh")

        self.register_buffer("temperature", torch.Tensor([1]))

    def update_temperature(self):
        self.temperature *= 0.965

    def forward(self, x):

        assert len(x.size()) == 5, "Only support multi slice input"

        N, S, C, H, W = x.size()

        weight = self.binarize(self.weight)

        if self.groups > 1:
            if S > self.groups:
                x = x[:, int((S - self.groups) // 2):int((S + self.groups) // 2)]
            elif S == self.groups:
                pass
            else:
                raise ValueError("The number of slices must be larger than groups ")

        elif self.groups == 1:
            x = x[:, S // 2].unsqueeze(1)

        else:
            raise ValueError("The number of groups must be larger than one ")

        out = F.conv2d(input=x.view(N, -1, H, W),
                       weight=weight,
                       bias=self.bias,
                       stride=self.stride,
                       padding=self.padding,
                       dilation=self.dilation,
                       groups=self.groups)

        return out


class XNORNet_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(XNORNet_BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.binary_activation1 = BinaryActivation(ste="Polynomial")
        self.binary_conv1 = conv3x3(inplanes, planes, stride=stride)

        self.bn2 = nn.BatchNorm2d(planes)
        self.binary_activation2 = BinaryActivation(ste="Polynomial")
        self.binary_conv2 = conv3x3(planes, planes, stride=1)

        self.relu = nn.ReLU()

        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = None

        self.stride = stride

        self.k_weight = Variable(torch.ones(1, 1, self.kernel_size, self.kernel_size).mul(1 / (self.kernel_size ** 2)))

    def get_scaling(self, x):
        k = x.mean(1, keepdim=True)
        if k.is_cuda:
            self.k_weight.cuda()
        k = F.conv(k, self.k_weight, padding=self.padding, stride=self.stride, dilation=dilation, groups=grpups)

        return k

    def forward(self, x):

        x = self.bn1(x)
        residual = x
        k_1 = self.get_scaling(x)

        x = self.binary_activation1(x)
        x = self.binary_conv1(x).mul(k_1)
        x = self.relu(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        y = self.bn2(y)
        y += residual
        k_2 = self.get_scaling(y)

        y = self.binary_activation2(x)
        y = self.binary_conv2(y).mul(k_2)
        y = self.relu(y)
        return y


class BiRealNet_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BiRealNet_BasicBlock, self).__init__()

        self.binary_activation1 = BinaryActivation(ste="Polynomial")
        self.binary_conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.binary_activation2 = BinaryActivation(ste="Polynomial")
        self.binary_conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = None

        self.stride = stride

        # self.features = dict()

    def forward(self, x):

        # self.features['BiRealNet_input_dimension'] = x.size()

        residual = x
        x = self.binary_activation1(x)
        x = self.binary_conv1(x)
        x = self.bn1(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual

        y = self.binary_activation2(x)
        y = self.binary_conv2(y)
        y = self.bn2(y)
        y += x

        return y


class ReActNet_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ReActNet_BasicBlock, self).__init__()

        self.move1_0 = LearnableBias(inplanes)
        self.binary_activation1 = BinaryActivation(ste="Polynomial")
        self.binary_conv1 = HardBinaryConv(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.move1_1 = LearnableBias(planes)
        self.prelu1 = nn.PReLU(planes)
        self.move1_2 = LearnableBias(planes)

        self.move2_0 = LearnableBias(planes)
        self.binary_activation2 = BinaryActivation(ste="Polynomial")
        self.binary_conv2 = HardBinaryConv(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.move2_1 = LearnableBias(planes)
        self.prelu2 = nn.PReLU(planes)
        self.move2_2 = LearnableBias(planes)

        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(inplanes, planes, kernel_size=1, padding=0),
                nn.BatchNor2md(planes),
            )
        else:
            self.downsample = None

        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.move1_0(x)
        x = self.binary_activation1(x)
        x = self.binary_conv1(x)
        x = self.bn1(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual

        x = self.move1_1(x)
        x = self.prelu1(x)
        x = self.move1_2(x)

        y = self.move2_0(x)
        y = self.binary_activation2(y)
        y = self.binary_conv2(y)
        y = self.bn2(y)

        y += x

        y = self.move2_1(y)
        y = self.prelu2(y)
        y = self.move2_2(y)

        return y


class BoolNetV1_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, max_slices=1, downsample=None):
        super(BoolNetV1_BasicBlock, self).__init__()

        self.binary_conv1 = MultiBConv(inplanes, planes, 3, 1, 1, groups=max_slices)
        if stride == 2:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = lambda x: x

        self.bn1 = nn.BatchNorm2d(planes)
        self.binary_activation1 = GhostSign(planes, slices=max_slices)

        self.binary_conv2 = MultiBConv(planes, planes, 3, 1, 1, groups=max_slices)
        self.bn2 = nn.BatchNorm2d(planes)
        self.binary_activation2 = GhostSign(planes, slices=max_slices)

        if stride == 2:
            self.downsample = nn.Sequential(
                HardBinaryConv(inplanes, planes, kernel_size=1, stride=1, padding=0),
                nn.maxpool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(planes),
                BinaryActivation(ste="Hardtanh")
            )
        else:
            self.downsample = None

        self.stride = stride

    def forward(self, x):

        assert len(x.size()) == 5, "only support 5-D(N S C H W) input"

        residual = x

        x = self.binary_conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.binary_activation1(x)

        if self.downsample is not None:
            residual = self.downsample(residual.view(N, -1, H, w))

        y = self.binary_conv2(x)
        y = self.bn2(y)
        y = self.binary_activation2(y)

        return y


class BoolNetV2_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, max_slices=1, downsample=None):
        super(BoolNetV2_BasicBlock, self).__init__()

        if inplanes != planes:
            inplanes = 2 * inplanes

        self.inplanes = inplanes

        self.binary_conv1 = MultiBConv(inplanes, planes, 3, 1, 1, groups=max_slices)
        if stride == 2:
            self.maxpool = nn.MaxPool2d(kernel_size=3, strride=2, padding=1)
        else:
            self.maxpool = lambda x: x
        self.bn1 = nn.BatchNorm2d(planes)
        self.binary_activation1 = GhostSign(planes, slices=max_slices)

        self.binary_conv2 = MultiBConv(planes, planes, 3, 1, 1, groups=max_slices)
        self.bn2 = nn.BatchNorm2d(planes)
        self.binary_activation2 = GhostSign(planes, slices=max_slices)

        if stride == 2:
            self.downsample = nn.Sequential(
                HardBinaryConv(inplanes * max_slices, planes, kernel_size=1, padding=0, groups=max_slices),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = None

        self.stride = stride

    def forward(self, input):
        N, S, C, H, W = input.size()

        # assert C // self.inplanes == 2, "The channels of input feature should  be twice of the first binary convolution input channels !"

        # "channel split"
        if C // self.inplanes == 2:
            x = input[:, :, :C // 2].contiguous()
            residual = x
        else:
            x = input
            residual = input

        x = self.binary_conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.binary_activation1(x)

        if self.downsample is not None:
            residual = self.downsample(residual.view(N, -1, H, W))

        y = self.binary_conv2(x)
        y = self.bn2(y)
        y = self.binary_activation2(y)

        if C // self.inlanes == 2:
            z = input[:, :, :C // 2:].contiguous()
        else:
            z = residual

        y_1 = y.unsqueeze(2)
        y_2 = z.unsqueeze(2)

        # "channel concatenet"
        y = torch.cat((y_1, y_2), dim=2)

        # "channel shuffle"
        y = y.transpose(2, 3).contiguous().view(N, S, -1, H // self.stride, W // self.stride).contiguous()

        return y


class HardSign(nn.Module):
    def __init__(self, range=[-1, 1], progressive=False):
        super(HardSign, self).__init__()
        self.range = range
        self.progressive = progressive
        self.register_buffer("temperature", torch.ones(1))

    def adjust(self, x, scale=0.1):
        self.temperature.mul_(scale)

    def forward(self, x, scale=None):
        if scale == None:
            scale = torch.ones_like(x)

        replace = x.clamp(self.range[0], self.range[1]) + scale
        x = x.div(self.temperature.clamp(min=1e-8)).clamp(-1, 1)
        if not self.progressive:
            sign = x.sign() * scale
        else:
            sign = x * scale
        return (sign - replace).detach() + replace


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, groups=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = kernel_size // 2
        self.groups = groups
        self.number_of_weights = in_chn // groups * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn // groups, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn((self.shape)) * 0.001, requires_grad=True)
        # self.weight_bias = nn.Parameter(torch.zeros(out_chn, in_chn, 1, 1))
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x):
        self.weight.data.clamp_(-1.5, 1.5)
        real_weights = self.weight
        # scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        # scaling_factor = scaling_factor.detach()
        # real_weights = real_weights - real_weights.mean(-1, keepdim = True).mean(-2, keepdim = True).mean(-3, keepdim = True) + self.weight_bias
        binary_weights_no_grad = (real_weights / self.temperature.clamp(min=1e-8)).clamp(-1, 1)
        cliped_weights = real_weights  # .clamp(-1.5, 1.5)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding, groups=self.groups)

        return y


class SqueezeAndExpand(nn.Module):
    def __init__(self, channels, planes, ratio=8, attention_mode="hard_sigmoid"):
        super(SqueezeAndExpand, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, channels // ratio, kernel_size=1, padding=0),
            nn.ReLU(channels // ratio),
            nn.Conv2d(channels // ratio, planes, kernel_size=1, padding=0),
        )

        if attention_mode == "sigmoid":
            self.attention = nn.Sigmoid()

        elif attention_mode == "hard_sigmoid":
            self.attention = HardSigmoid()

        else:
            self.attention = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.se(x)
        x = self.attention(x)
        return x


class Attention(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.1, gamma=1e-6, groups=1):
        super(Attention, self).__init__()

        self.inplanes = inplanes
        self.planes = planes

        self.move = LearnableBias(inplanes)
        self.binary_activation = HardSign(range=[-1.5, 1.5])
        self.binary_conv = HardBinaryConv(inplanes, planes, kernel_size=3, stride=stride, groups=groups)

        self.norm1 = nn.BatchNorm2d(planes)
        self.norm2 = nn.BatchNorm2d(planes)

        self.activation1 = nn.PReLU(inplanes)
        self.activation2 = nn.PReLU(planes)

        self.downsample = downsample
        self.stride = stride
        if stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)

        self.se = SqueezeAndExpand(planes, planes, attention_mode="sigmoid")
        
    def forward(self, input):

        input = self.activation1(input)
        
        residual = input

        if self.stride == 2:
            residual = self.pooling(residual)

        x = self.move(input)
        x = self.binary_activation(x, scale=None)
        x = self.binary_conv(x)
        x = self.norm1(x)
        x = self.activation2(x)
        x = self.se(residual) * x

        x = x * residual
        
        x = self.norm2(x)
        x = x + residual

        return x


class FFN_3x3(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.1, gamma=1e-8, groups=1):
        super(FFN_3x3, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride

        self.move = LearnableBias(inplanes)
        self.binary_activation = HardSign(range=[-1.5, 1.5])
        self.binary_conv = HardBinaryConv(inplanes, planes, kernel_size=3, stride=stride, groups=groups)

        self.norm1 = nn.BatchNorm2d(planes)
        self.norm2 = nn.BatchNorm2d(planes)

        self.activation1 = nn.PReLU(planes)

        if stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)

        self.se = SqueezeAndExpand(inplanes, planes, attention_mode="sigmoid")
        
    def forward(self, input):

        residual = input

        if self.stride == 2:
            residual = self.pooling(residual)

        x = self.move(input)
        x = self.binary_activation(x, scale=None)
        x = self.binary_conv(x)
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.se(residual) * x

        x = self.norm2(x)
        x = x + residual

        return x


class FFN_1x1(nn.Module):
    def __init__(self, inplanes, planes, stride=1, attention=True, drop_rate=0.1, gamma=1e-3):
        super(FFN_1x1, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride

        self.move = LearnableBias(inplanes)
        self.binary_activation = HardSign(range=[-1.5, 1.5])
        self.binary_conv = HardBinaryConv(inplanes, planes, kernel_size=1, stride=stride, padding=0)

        self.norm1 = nn.BatchNorm2d(planes)
        self.norm2 = nn.BatchNorm2d(planes)

        self.activation1 = nn.PReLU(planes)

        self.dropout_pre = nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity()
        self.dropout_aft = nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity()

        if stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)

        self.attention = attention

        if attention:
            self.se = SqueezeAndExpand(inplanes, planes, attention_mode="sigmoid")


    def forward(self, input):

        residual = input
        
        if self.stride == 2:
            residual = self.pooling(residual)

        x = self.move(input)
        x = self.binary_activation(x, scale=None)
        x = self.binary_conv(x)
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.se(residual)*x

        x = self.norm2(x)
        x = x + residual

        return x


class BNext_BasicModule(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.1, mode="scale"):
        super(BNext_BasicModule, self).__init__()
        self.inplanes = inplanes
        self.planes = planes

        if mode == "scale":
            self.Attention = Attention(inplanes, inplanes, stride, None, drop_rate=drop_rate, groups=1)
        
        else:
            self.Attention = FFN_3x3(inplanes, inplanes, stride, None, drop_rate=drop_rate, groups=1)

        if inplanes == planes:
            self.FFN = nn.Sequential(FFN_1x1(inplanes, inplanes, drop_rate=drop_rate))

        else:
            self.FFN_1 = nn.Sequential(FFN_1x1(inplanes, inplanes, drop_rate=drop_rate))

            self.FFN_2 = nn.Sequential(FFN_1x1(inplanes, inplanes, drop_rate=drop_rate))

    def forward(self, input):
        x = self.Attention(input)

        if self.inplanes == self.planes:
            y = self.FFN(x)

        else:
            y_1 = self.FFN_1(x)
            y_2 = self.FFN_2(x)
            y = torch.cat((y_1, y_2), dim=1)

        return y


class BNext_BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride, downsample, drop_rate):
        super(BNext_BasicBlock, self).__init__()
        self.Attention = BNext_BasicModule(inplanes, planes, stride, None, drop_rate = drop_rate, mode = "scale")
        self.FFN = BNext_BasicModule(planes, planes, 1, None, drop_rate = drop_rate, mode = "bias")

    def forward(self, input):
        x = self.Attention(input)
        y = self.FFN(x)

        return y


class Architecture(nn.Module):
    def __init__(self, arc="BiRealNet", inplanes=64, out_planes=64, stride=1, max_slices=1):
        super(Architecture, self).__init__()

        assert arc in {"BiRealNet", "ReActNet", "BoolNetV1",
                       "BoolNetV2", "BNext"}, "Only support {BiRealNet, ReActNet, BoolNetV1, BoolNetV2, BNext}, but got {}".format(arc)

        self.arc = arc

        if self.arc == "BiRealNet":
            self.basicblock = BiRealNet_BasicBlock(inplanes=inplanes, planes=out_planes, stride=stride)

        elif self.arc == "ReActNet":
            self.basicblock = ReActNet_BasicBlock(inplanes=inplanes, planes=out_planes, stride=stride)

        elif self.arc == "BoolNetV1":
            self.basicblock = BoolNetV1_BasicBlock(inplanes=inplanes, planes=out_planes, stride=stride, max_slices=max_slices)
        
        elif self.arc == "BoolNetV2":
            self.basicblock = BoolNetV2_BasicBlock(inplanes=inplanes, planes=out_planes, stride=stride, max_slices=max_slices)

        elif self.arc == "XNORNet":
            self.basciblock = BasicBlock(inplanes=inplanes, planes=out_planes, stride=stride)

        elif self.arc == "BNext":
            self.basicblock = BNext_BasicBlock(inplanes=inplanes, planes=out_planes, stride=stride, downsample = None, drop_rate = 0.1)
                      
                                
    def forward(self, x):
        return self.basicblock(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example of Architecture Details")
    parser.add_argument("--arc", default="BiRealNet", help="specifing the architecture to be used")
    parser.add_argument("--inplanes", default=64, type=int, help="specifing the inplanes of basicblock")
    parser.add_argument("--out_planes", default=64, type=int, help="specifing the out_planes of basicblock")
    parser.add_argument("--mode", default="train", type=str, help="specifing the computation type of neural network")
    parser.add_argument("--max_slices", default=1, type=int,
                        help="specifing the number of slices to be used in MultiSlice strategy")
    parser.add_argument("--stride", default=1, type=int, help="using reduce residual shortcut if stride == 2")

    opt = parser.parse_args()
    model = Architecture(arc=opt.arc, inplanes=opt.inplanes, out_planes=opt.out_planes, stride=opt.stride,
                         max_slices=opt.max_slices)

    if opt.mode == "train":
        model.train()

    else:
        model.eval()

    writer = SummaryWriter()

    if not opt.arc in {"BoolNetV1", "BoolNetV2"}:
        input_tensor = torch.randn(1, opt.inplanes, 32, 32)

    elif opt.arc == "BoolNetV2":
        input_tensor = torch.randn(1, opt.max_slices, 2 * opt.inplanes, 32, 32)

    else:
        input_tensor = torch.randn(1, opt.max_slices, opt.inplanes, 32, 32)

    out_tensor = model(input_tensor)

    #summary(model, input_tensor)
    writer.add_graph(model, input_tensor)

    writer.close()

