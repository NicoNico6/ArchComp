import argparse
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

__all__ = ['birealnet', 'reactnet', 'boolnetv1', 'boolnetv2']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def OR(x, y):                                  #-1,1
    """Logic OR"""
    y = y.add(1).div(2)                        # 0,1
    x = x.add(1).div(2)                        # 0,1
    return x.add(y).clamp(0,1).mul(2).add(-1)
    
def XNOR(x, y):                                #-1,1
    """Logic XNOR"""
    y = x.mul(y)
    return y


class BinaryActivation(nn.Module):
    def __init__(self, ste = "Hardtanh"):
        super(BinaryActivation, self).__init__()
        self.ste = ste
        assert self.ste in {"Hardtanh", "Polynomial"}
        
    def polynomial_forward(self, x):
        out_forward = torch.sign(x)
        if not self.training:
          return out_forward
          
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
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
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, groups = 1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.number_of_weights = in_chn // groups * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn//groups, kernel_size, kernel_size)
        #self.weight = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)

    def forward(self, x):
        
        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()

        if not self.training:
          binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
          
          return  F.conv2d(x, binary_weights_no_grad, stride=self.stride, padding=self.padding, groups = self.groups)
  
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding, groups = self.groups)

        return y

class BiRealNet_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BiRealNet_BasicBlock, self).__init__()

        self.binary_activation1 = BinaryActivation(ste = "Polynomial")
        self.binary_conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.binary_activation2 = BinaryActivation(ste = "Polynomial")
        self.binary_conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride

        #self.features = dict()
        
    def forward(self, x):
        
        #self.features['BiRealNet_input_dimension'] = x.size()
        
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
        self.binary_activation1 = BinaryActivation(ste = "Polynomial")
        self.binary_conv1 = HardBinaryConv(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.move1_1 = LearnableBias(planes)
        self.prelu1 = nn.PReLU(planes)
        self.move1_2 = LearnableBias(planes)

        self.move2_0 = LearnableBias(planes)
        self.binary_activation2 = BinaryActivation(ste = "Polynomial")
        self.binary_conv2 = HardBinaryConv(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.move2_1 = LearnableBias(planes)
        self.prelu2 = nn.PReLU(planes)
        self.move2_2 = LearnableBias(planes)

        self.downsample = downsample
        self.stride = stride
        
        #self.features = dict()
        
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BoolNetV1_BasicBlock, self).__init__()

        self.binary_conv1 = HardBinaryConv(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.binary_activation1 = BinaryActivation(ste = "Hardtanh")
        
        self.binary_conv2 = HardBinaryConv(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.binary_activation2 = BinaryActivation(ste = "Hardtanh")
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        
        
        residual = x
        
        x = self.binary_conv1(x)
        x = self.bn1(x)
        x = self.binary_activation1(x)
        
        if self.downsample is not None:
            residual = self.downsample(residual)

        x = XNOR(x, residual)
        
        y = self.binary_conv2(x)
        y = self.bn2(y)
        y = self.binary_activation2(y)
        
        y = OR(y, x)
        
        return y

class BoolNetV2_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BoolNetV2_BasicBlock, self).__init__()
        self.inplanes = inplanes
        
        self.binary_conv1 = HardBinaryConv(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.global_zero_points1 = nn.Sequential(
                                    nn.BatchNorm2d(planes),
                                    BinaryActivation(ste = "Hardtanh"),
                                    HardBinaryConv(planes, planes, groups = planes),
                                    nn.BatchNorm2d(planes),
                                    nn.AdaptiveMaxPool2d((1,1))
                                  )                  
        self.binary_activation1 = BinaryActivation(ste = "Hardtanh")
        
        self.binary_conv2 = HardBinaryConv(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.global_zero_points2 = nn.Sequential(
                                    nn.BatchNorm2d(planes),
                                    BinaryActivation(ste = "Hardtanh"),
                                    HardBinaryConv(planes, planes, groups = planes),
                                    nn.BatchNorm2d(planes),
                                    nn.AdaptiveMaxPool2d((1,1)) 
                                  )
        self.binary_activation2 = BinaryActivation(ste = "Hardtanh")
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, input):
        N, C, H, W = input.size()
        
        #assert C // self.inplanes == 2, "The channels of input feature should  be twice of the first binary convolution input channels !"
        
        #"channel split"
        x = input[:, :C//2].contiguous()
        residual = x
        z = input[:, C//2:].contiguous()
        
        x = self.binary_conv1(x)
        x = self.bn1(x).add(self.global_zero_points1(x))
        x = self.binary_activation1(x)
        
        if self.downsample is not None:
            residual = self.downsample(residual)

        x = XNOR(x, residual)
        
        y = self.binary_conv2(x)
        y = self.bn2(y).add(self.global_zero_points2(x))
        y = self.binary_activation2(y)
        
        y_1 = OR(y, residual).unsqueeze(1)
        y_2 = z.unsqueeze(1)
        
        #"channel concatenet"
        y = torch.cat((y_1, y_2), dim = 1)  
        
        #"channel shuffle"
        y = y.transpose(1,2).contiguous().view(N, -1, H, W).contiguous()
        
        return y

class Architecture(nn.Module):
  def __init__(self, arc = "BiRealNet", inplanes = 64, out_planes = 64):
    super(Architecture, self).__init__()
    
    assert arc in {"BiRealNet", "ReActNet", "BoolNetV1", "BoolNetV2"}, "Only support {BiRealNet, ReActNet, BoolNetV1, BoolNetV2}, but got {}".format(arc) 
    
    self.arc = arc
    
    if self.arc == "BiRealNet":
      self.basicblock = BiRealNet_BasicBlock(inplanes = inplanes, planes = out_planes)
    
    elif self.arc == "ReActNet":
      self.basicblock = ReActNet_BasicBlock(inplanes = inplanes, planes = out_planes)
    
    elif self.arc == "BoolNetV1":
      self.basicblock = BoolNetV1_BasicBlock(inplanes = inplanes, planes = out_planes)
    
    elif self.arc == "BoolNetV2":
      self.basicblock = BoolNetV2_BasicBlock(inplanes = inplanes, planes = out_planes) 
  
  def forward(self, x):
    return self.basicblock(x)
        
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Example of Architecture Details")
  parser.add_argument("--arc", default="BiRealNet", help="specifing the architecture to be used")
  parser.add_argument("--inplanes", default=64, type = int, help="specifing the inplanes of basicblock")
  parser.add_argument("--out_planes", default=64, type = int, help="specifing the out_planes of basicblock")
  parser.add_argument("--mode", default="train", type = str, help="specifing the computation type of neural network")

  opt = parser.parse_args()
  model = Architecture(arc = opt.arc, inplanes = opt.inplanes, out_planes = opt.out_planes)
  
  if opt.mode == "train":
    model.train()
  
  else:
    model.eval()

  writer = SummaryWriter()
  
  if not opt.arc == "BoolNetV2":
    input_tensor = torch.randn(1, opt.inplanes, 32, 32)
  
  else:
    input_tensor = torch.randn(1, 2*opt.inplanes, 32, 32)

  out_tensor = model(input_tensor)

  writer.add_graph(model, input_tensor)
  
  writer.close()
  
