from torch import nn
import torch
from skimage.filters import gabor_kernel
from skimage.transform import resize
from scipy import optimize
import numpy as np

class L2Pool(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(*args, **kwargs)

        self.kernel_size = self.pool.kernel_size
        self.stride = self.pool.stride
        self.padding = self.pool.padding
        self.ceil_mode = self.pool.ceil_mode
        self.count_include_pad = self.pool.count_include_pad
        self.divisor_override = self.pool.divisor_override
        
    def forward(self, x):
        return torch.sqrt(self.pool(x ** 2))

def _resize(inp, shape, method='pad_crop', fill_value=None):
    if fill_value is None:
        fill_value=0
    if method == 'pad_crop':
        res = np.full(shape, fill_value)
        start_x=0
        start_y=0
        if inp.shape[0] < shape[0]:
            start_x = (shape[0]-inp.shape[0])//2
        else:
            diff = inp.shape[0] - shape[0]
            inp=inp[diff//2:diff//2+shape[0]]

        if inp.shape[1] < shape[1]:
            start_y = (shape[1]-inp.shape[1])//2
        else:
            diff = inp.shape[1] - shape[1]
            inp=inp[:,diff//2:diff//2+shape[1]]

        res[start_x:start_x+inp.shape[0], start_y:start_y+inp.shape[1]] = inp
    else:
        res = resize(inp, shape)
    return res

def _gabor_fit_mse_loss(params, image, sigma_x, sigma_y, n_stds):
    kernel = params[3]*gabor_kernel(frequency=1.0/params[0], theta=params[1], sigma_x=sigma_x, sigma_y=sigma_y, n_stds=n_stds, offset=params[2]).real + params[4]
    resulting_kernel = _resize(kernel, image.shape, fill_value=params[4])
    mse = torch.mean(torch.abs(image - resulting_kernel)**2)
    return mse

def fit_gabor_filter(image, wavelength=4, theta=1, phase_offset=0,  maxiter=100):
    amplitude, x0, y0, sigma_x, sigma_y, offset = fit_gaussian_2d(np.abs(image))

    sign = np.sign(image[image.shape[0]//2, image.shape[1]//2])
    sigma = max(sigma_x, sigma_y)
    initial_params = (wavelength, theta, phase_offset, sign*amplitude, offset)
    bounds=[(4,image.shape[0]*4), (0,np.pi), (0, 2*np.pi), (None, None), (None, None)]
    result = optimize.minimize(_gabor_fit_mse_loss, initial_params, args=(image, sigma, sigma, 4), bounds=bounds, method='Nelder-Mead', options={'maxiter':maxiter})
    params = result.x
    reproduced = params[3]*gabor_kernel(frequency=1.0/params[0], theta=params[1], sigma_x=sigma, sigma_y=sigma, n_stds=4, offset=params[2]).real+params[4]
    reproduced = _resize(reproduced, image.shape, fill_value=params[4])
    return reproduced, result.fun, result.x


def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, offset):
    x, y = xy
    exp_term = -((x - xo) ** 2 / (2 * sigma_x ** 2) + (y - yo) ** 2 / (2 * sigma_y ** 2))
    return amplitude * np.exp(exp_term) + offset

def fit_gaussian_2d(image, amplitude=1, x0=None, y0=None, sigma_x=2, sigma_y=2, offset=0):
    if x0 is None:
        x0= image.shape[0]//2
    if y0 is None:
        y0= image.shape[1]//2

    # Generate meshgrid
    x, y = np.meshgrid(np.linspace(0,image.shape[0], image.shape[0]), np.linspace(0,image.shape[1], image.shape[1]))

    initial_guess = (amplitude, x0, y0, sigma_x, sigma_y, offset)

    params, covariance = optimize.curve_fit(gaussian_2d, (x.ravel(), y.ravel()), image.ravel(), p0=initial_guess)

    return params