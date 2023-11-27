from torch import nn
import torch
from skimage.transform import resize
from scipy import optimize
import numpy as np
from scipy.ndimage import gaussian_filter

import os
import fnmatch

def find_files_in_folder(folder, partial_name):
    matching_files = []
    
    for root, dirs, files in os.walk(folder):
        for filename in fnmatch.filter(files, f'*{partial_name}*'):
            matching_files.append(os.path.join(root, filename))
    matching_files.sort()
    return matching_files

def normalize(img):
    return (img-img.mean())/(img.max()-img.min())

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

def gabor_kernel(shape=(128, 128), frequency=0.2, theta=0, sigma_x=1, sigma_y=1, phase_offset=0, center_offset=(0, 0), factor=1, offset=0):
    """
    Generate a Gabor kernel with the specified shape, center offset, and phase offset.

    Parameters:
        shape (tuple, optional): Desired output shape (height, width) of the Gabor kernel. Default is (128, 128).
        frequency (float, optional): Spatial frequency of the sinusoidal component. Default is 0.2.
        theta (float, optional): Orientation of the filter (in radians). Default is 0.
        sigma_x (float, optional): Standard deviation of the Gaussian envelope in the x-direction. Default is 1.
        sigma_y (float, optional): Standard deviation of the Gaussian envelope in the y-direction. Default is 1.
        phase_offset (float, optional): Phase offset of the Gabor kernel (in radians). Default is 0.
        center_offset (tuple, optional): Center offset of the Gaussian envelope (offset_x, offset_y). Default is (0, 0).
        factor (float, optional): scale the resulting filters values
        offset  (float, optional): added to the resulting filters values

    Returns:
        numpy.ndarray: Gabor kernel with the specified shape, center offset, and phase offset.
    """
    x, y = np.meshgrid(np.linspace(-shape[1]/2, shape[1]/2, shape[1]), np.linspace(-shape[0]/2, shape[0]/2, shape[0]))
    
    # Apply center offset to the coordinates
    x -= center_offset[0]
    y -= center_offset[1]
    
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    
    envelope = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2))
    sinusoid = np.cos(2 * np.pi * frequency * x_theta + phase_offset)
    
    gabor_kernel = envelope * sinusoid
    return gabor_kernel

def _gabor_fit_mse_loss(params, image, theta, sigma_x, sigma_y):
    resulting_kernel = gabor_kernel(image.shape, frequency=1.0/params[0], theta=theta, sigma_x=sigma_x, sigma_y=sigma_y, phase_offset=params[1], factor = params[2], offset = params[3])
    # resulting_kernel = _resize(kernel, image.shape, fill_value=params[3])
    mse = np.mean((image - resulting_kernel)**2)
    return mse

def fit_gabor_filter(image, wavelength=None, theta=None, phase_offset=0,  maxiter=100):
    if theta is None:
        theta = detect_angle(image)
    if wavelength is None:
        _min = np.array(np.unravel_index(image.real.argmin(), image.real.shape))
        _max = np.array(np.unravel_index(image.real.argmax(), image.real.shape))
        wavelength = np.sqrt(np.sum((_max-_min)**2))*2

    amplitude, x0, y0, sigma_x, sigma_y, offset = _fit_gaussian_2d(np.abs(image), theta=-theta, maxiter=maxiter) # theta flip bc gaussian kernel rotates mathematically positive (anticlockwise), but gabor kernel defined differently 

    sign = np.sign(image[image.shape[0]//2, image.shape[1]//2])
    initial_params = (np.clip(wavelength, 4, image.shape[0]*3), phase_offset, sign*amplitude, offset)
    bounds=[(4, image.shape[0]*3), (-np.pi, np.pi), (None, None), (None, None)]
    result = optimize.minimize(_gabor_fit_mse_loss, initial_params, args=(image, theta, sigma_x, sigma_y), bounds=bounds, method='Nelder-Mead', options={'maxiter':maxiter})
    params = result.x
    reproduced = gabor_kernel(image.shape, frequency=1.0/params[0], theta=theta, sigma_x=sigma_x, sigma_y=sigma_y, phase_offset=params[1], factor = params[2], offset = params[3])
    reproduced = _resize(reproduced, image.shape, fill_value=params[3])
    return reproduced, result.fun, params

def detect_angle(image, n_thetas=180):
    tmp_image = np.copy(image)
    tmp_image[np.abs(image)<np.quantile(np.abs(image), 0.9)]=0
    thetas = (np.array([i+0.5 for i in range(n_thetas)])/n_thetas*2*np.pi - np.pi)/2
    h, theta, d = weighted_hough_line(tmp_image, theta=thetas)
    return thetas[np.argmax((h**2).sum(axis=0))]

def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset, theta):
    x, y = xy
    a = np.cos(theta) ** 2 / (2 * sigma_x ** 2) + np.sin(theta) ** 2 / (2 * sigma_y ** 2)
    b = -np.sin(2 * theta) / (4 * sigma_x ** 2) + np.sin(2 * theta) / (4 * sigma_y ** 2)
    c = np.sin(theta) ** 2 / (2 * sigma_x ** 2) + np.cos(theta) ** 2 / (2 * sigma_y ** 2)

    exp_term = a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2
    return amplitude * np.exp(-exp_term) + offset

def gaussian_kernel(shape=(128, 128), amplitude=1, x0=None, y0=None, sigma_x=1, sigma_y=1, offset=0, theta=0):
    if x0 is None:
        x0 = shape[0]/2
    if y0 is None:
        y0 = shape[1]/2
    x, y = np.meshgrid(np.linspace(0,shape[0], shape[0]), np.linspace(0,shape[1], shape[1]))
    gauss = gaussian_2d((x,y), amplitude, x0, y0, sigma_x, sigma_y, offset, theta)
    return gauss

def _gaussian_fit_mse_loss(params, image, theta):
    amplitude, x0, y0, sigma_x, sigma_y, offset = params
    resulting_kernel = gaussian_kernel(image.shape, amplitude, x0, y0, sigma_x, sigma_y, offset, theta)
    # resulting_kernel = _resize(kernel, image.shape, fill_value=params[3])
    mse = np.mean((image - resulting_kernel)**2)
    return mse

def _fit_gaussian_2d(image, amplitude=1, x0=None, y0=None, sigma_x=2, sigma_y=2, offset=0, theta=0, blur_sigma=0, maxiter=100):
    if x0 is None:
        x0 = image.shape[0]/2
    if y0 is None:
        y0 = image.shape[1]/2

    _img=gaussian_filter(np.abs(image), sigma=blur_sigma)
    
    initial_params = (amplitude, x0, y0, sigma_x, sigma_y, offset)
    bounds=[(0, None), (0, _img.shape[0]), (0, _img.shape[1]), (.1, None), (.1, None), (0,None)]
    result = optimize.minimize(_gaussian_fit_mse_loss, initial_params, args=(_img, theta), bounds=bounds, method='Nelder-Mead', options={'maxiter':maxiter})
    return result.x

def fit_gaussian_2d(image, amplitude=1, x0=None, y0=None, sigma_x=2, sigma_y=2, offset=0, theta=0, blur_sigma=0):
    if x0 is None:
        x0= image.shape[0]//2
    if y0 is None:
        y0= image.shape[1]//2
    
    _img=gaussian_filter(np.abs(image), sigma=blur_sigma)
        

    # Generate meshgrid
    x, y = np.meshgrid(np.linspace(0,_img.shape[0], _img.shape[0]), np.linspace(0,_img.shape[1], _img.shape[1]))

    initial_guess = (amplitude, x0, y0, sigma_x, sigma_y, offset, theta)
    try:
        params, covariance = optimize.curve_fit(gaussian_2d, (x.ravel(), y.ravel()), _img.ravel(), p0=initial_guess)
    except:
        params = initial_guess

    return params

def weighted_hough_line(img: np.ndarray,
                theta: np.ndarray):
    """Perform a straight line Hough transform.
    In difference to the original hough line transform, the value of each pixel is its 'voting power'.

    Parameters
    ----------
    img : (M, N) ndarray
        Input image with nonzero values representing edges.
    theta : 1D ndarray of float6411
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