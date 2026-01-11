#code from augmix

import numpy as np
import random
import torch
import torchvision.transforms.functional as TF
import torchvision
import torch.nn.functional as F


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

def autocontrast(img, _level):
    minimum = img.amin(dim=(-2, -1), keepdim=True)
    maximum = img.amax(dim=(-2, -1), keepdim=True)
    scale = 1 / (maximum - minimum)
    eq_idxs = torch.isfinite(scale).logical_not()
    minimum[eq_idxs] = 0
    scale[eq_idxs] = 1

    return ((img - minimum) * scale).clamp(0, 1).to(img.dtype)


def _scale_channel(img_chan) :
    img = img_chan.clamp(0, 1)
    hist = torch.histc(img, bins=256, min=0.0, max=1.0)
    cdf = hist.cumsum(0)
    cdf = cdf / cdf[-1]

    bin_edges = torch.linspace(0, 1, 256 + 1, device=img.device)

    img_flat = img.reshape(-1)
    bin_idx = torch.bucketize(img_flat, bin_edges, right=True) - 1
    bin_idx = bin_idx.clamp(0, 256 - 1)
    img_eq = cdf[bin_idx]

    return img_eq.reshape(img.shape)


def equalize(img, _level):
    return torch.stack([_scale_channel(img[c]) for c in range(img.size(0))])

def posterize(img, level):
    bits = 4 - int_parameter(sample_level(level), 4)
    img = (img*255.0).to(torch.uint8)

    mask = -int(2 ** (8 - bits))
    return (img & mask).to(torch.float32) /255.0 


def rotate(img, level):
    degrees = int_parameter(sample_level(level), 30)
    if random.random() > 0.5:
        degrees = -degrees
    return TF.rotate(img, degrees, interpolation=TF.InterpolationMode.BILINEAR)

def solarize(img, level):
    threshold = 256 - int_parameter(sample_level(level), 256)
    return torch.where(img * 255 > threshold,
                       1.0 - img,
                       img)

def shear_x(img, level):
    shear = float_parameter(sample_level(level), 0.3)
    if random.random() > 0.5:
        shear = -shear
    shear = shear * 180 / torch.pi
    return TF.affine(img, angle=0, translate=[0,0], scale=1.0,
                     shear=[shear, 0.0],
                     interpolation=TF.InterpolationMode.BILINEAR)

def shear_y(img, level):
    shear = float_parameter(sample_level(level), 0.3)
    if random.random() > 0.5:
        shear = -shear
    shear = shear * 180 / torch.pi
    return TF.affine(img, angle=0, translate=[0,0], scale=1.0,
                     shear=[0.0, shear],
                     interpolation=TF.InterpolationMode.BILINEAR)



def translate_x(img, level):
    shift = int_parameter(sample_level(level), img.shape[-1]/ 3)
    if random.random() > 0.5:
        shift = -shift
    return TF.affine(img, angle=0, translate=[shift, 0], scale=1.0,
                     shear=[0.0, 0.0],
                     interpolation=TF.InterpolationMode.BILINEAR)

def translate_y(img, level):
    shift = int_parameter(sample_level(level), img.shape[-2]/ 3)
    if random.random() > 0.5:
        shift = -shift
    return TF.affine(img, angle=0, translate=[0, shift], scale=1.0,
                     shear=[0.0, 0.0],
                     interpolation=TF.InterpolationMode.BILINEAR)


augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

#precomputed means of folds
MEANS=[[0.0943, 0.0905, 0.1178, 0.0836, 0.0655],[0.0989, 0.0930, 0.1249, 0.0856, 0.0672], 
       [0.0978, 0.0919, 0.1258, 0.0831, 0.0675],[0.0959, 0.0923, 0.1205, 0.0925, 0.0671],
       [0.0967, 0.0941, 0.1208, 0.0896, 0.0687]]
STDS=[[0.1187, 0.1094, 0.1123, 0.0848, 0.0982], [0.1217, 0.1116, 0.1148, 0.0859, 0.1011],
      [0.1211, 0.1101, 0.1161, 0.0841, 0.0999], [0.1186, 0.1111, 0.1126, 0.0881, 0.0995],
      [0.1193, 0.1122, 0.1128, 0.0868, 0.0995]]


def normalize(image, fold_id):
  """Normalize input image channel-wise to zero mean and unit variance."""
  normalize = torchvision.transforms.Normalize(
    mean=MEANS[fold_id],
    std=STDS[fold_id]
  )

  return normalize(image)


def apply_op(image, op, severity):
    image = image.clamp(0.0, 1.0)
    augmented = op(image, severity)

    return augmented.clamp(0.0, 1.0)



def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1., fold_id = None, ablation=None):
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
  augs = augmentations.copy()
  if ablation!=None:
     augs.pop(ablation)
  
  ws = torch.distributions.Dirichlet(torch.ones(width) * alpha).sample().to(image.device)
  m  = torch.distributions.Beta(alpha, alpha).sample().to(image.device)

  img_pre = normalize(image, fold_id)
  mix = torch.zeros_like(image)
  for i in range(width):
    image_aug = image.clone()
    d = depth if depth > 0 else np.random.randint(1, 4)
    for _ in range(d):
      op = np.random.choice(augs)
      image_aug = apply_op(image_aug, op, severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * normalize(image_aug, fold_id)

  mixed = m * img_pre + (1 - m) * mix

  return mixed



class GaussianNoise:
    def __init__(self, std=2):
        self.std = std

    def __call__(self, x):
        return x + torch.randn_like(x) * self.std


class PoissonNoise:
    def __call__(self, x):
        # assumes x >= 0
        x = torch.clamp(x, min=0.0)
        return torch.poisson(x)


class GaussianBlur:
    def __init__(self, kernel_size=3, sigma=(0.1, 1.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(*self.sigma)
        return TF.gaussian_blur(x, self.kernel_size, [sigma, sigma])
class IntensityJitter:
    def __init__(self, brightness=0.15, contrast=0.15):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, x):
        # brightness
        b = 1.0 + random.uniform(-self.brightness, self.brightness)
        x = x * b

        # contrast (global, not per-channel)
        mean = x.mean(dim=(1, 2), keepdim=True)
        c = 1.0 + random.uniform(-self.contrast, self.contrast)
        x = (x - mean) * c + mean

        return x
    
class RandomOneOf:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        t1 = random.choice(self.transforms)
        t2 = random.choice(self.transforms)
        return t2(t1(x))
        return t1(x)
    
microscopy_aug = RandomOneOf([
    lambda x: TF.hflip(x),
    lambda x: TF.vflip(x),
    lambda x: TF.resized_crop(
        x,
        top=random.randint(0, int(0.15 * x.shape[1])),
        left=random.randint(0, int(0.15 * x.shape[2])),
        height=int(0.85 * x.shape[1]),
        width=int(0.85 * x.shape[2]),
        size=[250, 250],
        antialias=True
    ),
    IntensityJitter(brightness=0.15, contrast=0.15),
    GaussianNoise(std=0.01),
    PoissonNoise(),
    GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
])

def get_aug():
    return microscopy_aug