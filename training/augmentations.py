#code from augmix

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import random
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F

# ImageNet code should change this value
IMAGE_SIZE = 499

def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)

#new
def autocontrast(img, _level):
    # img: C×H×W in [0,1]
    cmin = img.amin(dim=(1,2), keepdim=True)
    cmax = img.amax(dim=(1,2), keepdim=True)
    scale = (cmax - cmin).clamp(min=1e-5)
    return (img - cmin) / scale

#new
def equalize(img, _level):
    # img: C×H×W in [0,1]
    C, H, W = img.shape
    out = torch.zeros_like(img)
    for c in range(C):
        flat = img[c].flatten()
        hist = torch.histc(flat, bins=256, min=0.0, max=1.0)
        cdf = hist.cumsum(0)
        cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-8)
        lut = (cdf * 255).clamp(0,255).to(torch.int64)
        vals = (flat * 255).to(torch.int64)
        out[c] = (lut[vals].reshape(H,W).float() / 255.0)
    return out

#new
def posterize(img, level):
    bits = 4 - int_parameter(sample_level(level), 4)
    shift = 8 - bits
    return torch.floor(img * 255 / (2**shift)) * (2**shift) / 255

#new
def rotate(img, level):
    degrees = int_parameter(sample_level(level), 30)
    if random.random() > 0.5:
        degrees = -degrees
    return TF.rotate(img, degrees, interpolation=TF.InterpolationMode.BILINEAR)

#new
def solarize(img, level):
    threshold = 256 - int_parameter(sample_level(level), 256)
    return torch.where(img * 255 > threshold,
                       1.0 - img,
                       img)
#new
def shear_x(img, level):
    shear = float_parameter(sample_level(level), 0.3)
    if random.random() > 0.5:
        shear = -shear
    return TF.affine(img, angle=0, translate=[0,0], scale=1.0,
                     shear=[shear, 0.0],
                     interpolation=TF.InterpolationMode.BILINEAR)

#new
def shear_y(img, level):
    shear = float_parameter(sample_level(level), 0.3)
    if random.random() > 0.5:
        shear = -shear
    return TF.affine(img, angle=0, translate=[0,0], scale=1.0,
                     shear=[0.0, shear],
                     interpolation=TF.InterpolationMode.BILINEAR)


#new
def translate_x(img, level):
    IMAGE_SIZE = img.shape[1]
    shift = int_parameter(sample_level(level), IMAGE_SIZE // 3)
    if random.random() > 0.5:
        shift = -shift
    return TF.affine(img, angle=0, translate=[shift, 0], scale=1.0,
                     shear=[0.0, 0.0],
                     interpolation=TF.InterpolationMode.BILINEAR)
#new
def translate_y(img, level):
    IMAGE_SIZE = img.shape[1]
    shift = int_parameter(sample_level(level), IMAGE_SIZE // 3)
    if random.random() > 0.5:
        shift = -shift
    return TF.affine(img, angle=0, translate=[0, shift], scale=1.0,
                     shear=[0.0, 0.0],
                     interpolation=TF.InterpolationMode.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(img, level):
    s = float_parameter(sample_level(level), 1.8) + 0.1
    mean = img.mean(dim=0, keepdim=True)
    return (img - mean) * s + mean


# operation that overlaps with ImageNet-C's test set
def contrast(img, level):
    s = float_parameter(sample_level(level), 1.8) + 0.1
    mean = img.mean()
    return (img - mean) * s + mean


# operation that overlaps with ImageNet-C's test set
def brightness(img, level):
    s = float_parameter(sample_level(level), 1.8) + 0.1
    return img * s


# operation that overlaps with ImageNet-C's test set
def sharpness(img, level):
    s = float_parameter(sample_level(level), 1.8) + 0.1

    # Laplacian kernel
    kernel = torch.tensor([
        [0, -1,  0],
        [-1, 5, -1],
        [0, -1,  0]
    ], dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0)

    C = img.shape[0]
    kernel = kernel.repeat(C, 1, 1, 1)
    img_blur = F.conv2d(img.unsqueeze(0), kernel, padding=1, groups=C)
    img_blur = img_blur.squeeze(0)

    return img * (1 - s) + img_blur * s


augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]

# CIFAR-10 constants
#TODO: change for 
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]


def normalize(image):
  """Normalize input image channel-wise to zero mean and unit variance."""
  image = image.transpose(2, 0, 1)  # Switch to channel-first
  mean, std = np.array(MEAN), np.array(STD)
  image = (image - mean[:, None, None]) / std[:, None, None]
  return image.transpose(1, 2, 0)


def apply_op(image, op, severity):
    image = image.clamp(0.0, 1.0)
    augmented = op(image, severity)

    return augmented.clamp(0.0, 1.0)


def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1., seed=1234):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.

  Returns:
    mixed: Augmented and mixed image.
  """
  np.random.seed(seed)
  torch.manual_seed(seed)
  
  ws = torch.distributions.Dirichlet(torch.ones(width) * alpha).sample().to(image.device)
  m  = torch.distributions.Beta(alpha, alpha).sample().to(image.device)

  mix = torch.zeros_like(image)
  for i in range(width):
    image_aug = image.clone()
    d = depth if depth > 0 else np.random.randint(1, 4)
    for _ in range(d):
      op = np.random.choice(augmentations)
      image_aug = apply_op(image_aug, op, severity)
    # Preprocessing commutes since all coefficients are convex
    #mix += ws[i] * normalize(image_aug)
    mix += ws[i] * image_aug

  #mixed = (1 - m) * normalize(image) + m * mix
  mixed = (1 - m) * image + m * mix
    # If you want it to *require* grad (but graph is broken before this point):
  #mixed.requires_grad_(True)
  return mixed


