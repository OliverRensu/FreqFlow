
from PIL import Image
from torchvision import transforms
from torch import fft
import torch

def gaussian_filter_high_pass(fshift, D):
    D=D/8.
    b, c, h, w = fshift.shape
    x = torch.arange(0, h)
    y = torch.arange(0, w)

    # Create a mesh grid
    x, y = torch.meshgrid(x, y, indexing='ij')
    #x, y = torch.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
    template = 1 - torch.exp(- dis_square / (2 * D ** 2)).cuda()
    return template.unsqueeze(0).unsqueeze(0).repeat(b,c,1,1) * fshift

def gaussian_filter_low_pass(fshift, D):
    D=D*2
    b, c, h, w = fshift.shape
    x = torch.arange(0, h)
    y = torch.arange(0, w)
    # Create a mesh grid
    x, y = torch.meshgrid(x, y, indexing='ij')
    center = (int((h - 1) / 2), int((w - 1) / 2))
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
    template = torch.exp(- dis_square / (2 * D ** 2)).cuda()
    return template.unsqueeze(0).unsqueeze(0).repeat(b,c,1,1) * fshift

def Fourier_filter(x, D):
    # FFT
    max_x, min_x = x.max(), x.min()
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    x_high = gaussian_filter_high_pass(x_freq, D)
    x_low = gaussian_filter_low_pass(x_freq, D)  
    x_high = fft.ifftshift(x_high, dim=(-2, -1))
    x_high = fft.ifftn(x_high, dim=(-2, -1)).real

    x_low = fft.ifftshift(x_low, dim=(-2, -1))
    x_low = fft.ifftn(x_low, dim=(-2, -1)).real
    return torch.clamp(x_low, min_x, max_x), torch.clamp(x_high, min_x, max_x)