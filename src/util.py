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

def _gabor_fit_mse_loss(params, image, theta, sigma_x, sigma_y, n_stds):
    kernel = params[2]*gabor_kernel(frequency=1.0/params[0], theta=theta, sigma_x=sigma_x, sigma_y=sigma_y, n_stds=n_stds, offset=params[1]).real + params[3]
    resulting_kernel = _resize(kernel, image.shape, fill_value=params[3])
    mse = np.mean((image - resulting_kernel)**2)
    return mse

def fit_gabor_filter(image, wavelength=4, theta=None, phase_offset=0,  maxiter=100):
    if theta is None:
        theta = detect_angle(image)
    amplitude, x0, y0, sigma_x, sigma_y, offset, gauss_theta = fit_gaussian_2d(np.abs(image), theta=theta)

    sign = np.sign(image[image.shape[0]//2, image.shape[1]//2])
    initial_params = (wavelength, phase_offset, sign*amplitude, offset)
    bounds=[(4,image.shape[0]*4), (0, 2*np.pi), (None, None), (None, None)]
    result = optimize.minimize(_gabor_fit_mse_loss, initial_params, args=(image, theta, sigma_x, sigma_y, 4), bounds=bounds, method='Nelder-Mead', options={'maxiter':maxiter})
    params = result.x
    reproduced = params[2]*gabor_kernel(frequency=1.0/params[0], theta=theta, sigma_x=sigma_x, sigma_y=sigma_y, n_stds=4, offset=params[1]).real+params[3]
    reproduced = _resize(reproduced, image.shape, fill_value=params[3])
    return reproduced, result.fun, result.x

def detect_angle(image, n_thetas=180):
    image[np.abs(image)<np.mean(np.abs(image))]=0
    thetas = (np.array([i+0.5 for i in range(n_thetas)])/n_thetas*2*np.pi - np.pi)/2
    h, theta, d = weighted_hough_line(image, theta=thetas)
    return thetas[np.argmax((h**2).sum(axis=0))]

def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, offset, theta):
    x, y = xy
    a = np.cos(theta) ** 2 / (2 * sigma_x ** 2) + np.sin(theta) ** 2 / (2 * sigma_y ** 2)
    b = -np.sin(2 * theta) / (4 * sigma_x ** 2) + np.sin(2 * theta) / (4 * sigma_y ** 2)
    c = np.sin(theta) ** 2 / (2 * sigma_x ** 2) + np.cos(theta) ** 2 / (2 * sigma_y ** 2)

    exp_term = a * (x - xo) ** 2 + 2 * b * (x - xo) * (y - yo) + c * (y - yo) ** 2
    return amplitude * np.exp(-exp_term) + offset

def fit_gaussian_2d(image, amplitude=1, x0=None, y0=None, sigma_x=2, sigma_y=2, offset=0, theta=0):
    if x0 is None:
        x0= image.shape[0]//2
    if y0 is None:
        y0= image.shape[1]//2

    # Generate meshgrid
    x, y = np.meshgrid(np.linspace(0,image.shape[0], image.shape[0]), np.linspace(0,image.shape[1], image.shape[1]))

    initial_guess = (amplitude, x0, y0, sigma_x, sigma_y, offset, theta)

    params, covariance = optimize.curve_fit(gaussian_2d, (x.ravel(), y.ravel()), image.ravel(), p0=initial_guess)

    return params

def weighted_hough_line(img: np.ndarray,
                theta: np.ndarray):
    """Perform a straight line Hough transform.
    In difference to the original hough line transform, the value of each pixel is its 'voting power'.

    Parameters
    ----------
    img : (M, N) ndarray
        Input image with nonzero values representing edges.
    theta : 1D ndarray of float64
        Angles at which to compute the transform, in radians.

    Returns
    -------
    H : (P, Q) ndarray of uint64
        Hough transform accumulator.
    theta : ndarray
        Angles at which the transform was computed, in radians.
    distances : ndarray
        Distance values.

    Notes
    -----
    The origin is the top left corner of the original image.
    X and Y axis are horizontal and vertical edges respectively.
    The distance is the minimal algebraic distance from the origin
    to the detected line.

    Examples
    --------
    Generate a test image:

    >>> img = np.zeros((100, 150), dtype=bool)
    >>> img[30, :] = 1
    >>> img[:, 65] = 1
    >>> img[35:45, 35:50] = 1
    >>> for i in range(90):
    ...     img[i, i] = 1
    >>> rng = np.random.default_rng()
    >>> img += rng.random(img.shape) > 0.95

    Apply the Hough transform:

    >>> out, angles, d = hough_line(img)

    .. plot:: hough_tf.py

    """
    # Compute the array of angles and their sine and cosine
    ctheta = np.cos(theta)
    stheta = np.sin(theta)

    # compute the bins and allocate the accumulator array
    offset = int(np.ceil(np.sqrt(img.shape[0] * img.shape[0] +
                                   img.shape[1] * img.shape[1])))
    max_distance = 2 * offset + 1
    accum = np.zeros((max_distance, theta.shape[0]))
    bins = np.linspace(-offset, offset, max_distance)

    # compute the nonzero indexes
    y_idxs, x_idxs = np.nonzero(img)

    nidxs = y_idxs.shape[0]  # x and y are the same shape
    nthetas = theta.shape[0]
    for i in range(nidxs):
        x = x_idxs[i]
        y = y_idxs[i]
        for j in range(nthetas):
            accum_idx = round((ctheta[j] * x + stheta[j] * y)) + offset
            accum[accum_idx, j] += img[y,x]

    return accum, theta, bins