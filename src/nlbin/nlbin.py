# Copyright 2025 Janik Haitz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file includes code from the kraken project,
# available at https://github.com/mittagessen/kraken and licensed under
# Apache 2.0 license https://github.com/mittagessen/kraken/blob/main/LICENSE.

import warnings
import logging

import numpy as np
from PIL import Image
from scipy.ndimage import affine_transform, percentile_filter, gaussian_filter, binary_dilation, zoom

try:
    import cupy as cp
    from cupyx.scipy.ndimage import affine_transform as cp_affine_transform
    from cupyx.scipy.ndimage import percentile_filter as cp_percentile_filter
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter
    from cupyx.scipy.ndimage import binary_dilation as cp_binary_dilation
    from cupyx.scipy.ndimage import zoom as cp_zoom
    GPU_AVAILABLE = True
except ModuleNotFoundError:
    GPU_AVAILABLE = False

logger = logging.getLogger("nlbin")
logging.getLogger("PIL").propagate = False


# This method is derived from nlbin Method (./kraken/binarization.py)
# in the kraken project, available at https://github.com/mittagessen/kraken.
def nlbin(im: Image.Image, threshold: float = 0.5, estimate_zoom: float = 0.5, estimate_scale: float = 1.0, 
          border: float = 0.1, percentile: int = 80, percentile_range: int = 20, low: int = 5,
          high: int = 90) -> tuple[Image.Image, Image.Image]:
    """
    Calculate a binary and a normalized image from a rgb image.
    Args:
        im: Input image.
        threshold: Binarization threshold.
        estimate_zoom: Zoom for page background estimation.
        estimate_scale: Scale for estimating a mask over the text region.
        border: Ignore this much of the border.
        percentile: Percentage for percentile filter.
        percentile_range: Range for percentile filter.
        low: Percentage for black estimation.
        high: Percentage for white estimation.
    Returns:
        A tuple containing the binary image and the normalized image.
    """
    logger.info(f"Processing file {im.filename}")
    logger.debug("Convert image to grayscale")
    im = im.convert('L')
    raw = np.array(im)

    logger.debug("Rescale image")
    raw = raw / float(np.iinfo(raw.dtype).max)

    logger.debug("Normalize image")
    if np.amax(raw) == np.amin(raw):
        raise ValueError("Input image is empty")
    im = raw - np.amin(raw)
    im /= np.amax(im)
    with warnings.catch_warnings():  # hide all warnings:
        warnings.simplefilter("ignore", UserWarning)
        m= zoom(im, estimate_zoom)
        m= percentile_filter(m, percentile, size=(percentile_range, 2))
        m= percentile_filter(m, percentile, size=(2, percentile_range))
        mh, mw = m.shape
        oh, ow = im.shape
        scale = np.diag([mh * 1.0 / oh, mw * 1.0 / ow])
        m = affine_transform(m, scale, output_shape=im.shape)  # rescale to original size
    w, h = np.minimum(np.array(im.shape), np.array(m.shape))
    flat = np.clip(im[:w, :h] - m[:w, :h] + 1, 0, 1)

    logger.debug("Estimate low and high thresholds")
    d0, d1 = flat.shape
    o0, o1 = int(border * d0), int(border * d1)
    est = flat[o0:d0 - o0, o1:d1 - o1]

    # by default, we use only regions that contain significant variance;
    # this makes the percentile based low and high estimates more reliable.
    v = est - gaussian_filter(est, estimate_scale * 20.0)
    v = gaussian_filter(v ** 2, estimate_scale * 20.0) ** 0.5
    v = (v > 0.3 * np.amax(v))
    v = binary_dilation(v, structure=np.ones((int(estimate_scale * 50), 1)))
    v = binary_dilation(v, structure=np.ones((1, int(estimate_scale * 50))))
    est = est[v]
    lo = np.percentile(est.ravel(), low)
    hi = np.percentile(est.ravel(), high)
    flat -= lo
    flat /= (hi - lo)
    flat = np.clip(flat, 0, 1)

    logger.debug("Create binary and normalized images")
    bin_im = np.array(255 * (flat > threshold), 'B')
    bin_im = Image.frombytes("L", (bin_im.shape[1], bin_im.shape[0]), bin_im.tobytes())
    nrm_im = Image.fromarray((flat * 255).astype(np.uint8))
    return bin_im, nrm_im


# This method is derived from nlbin Method (./kraken/binarization.py)
# in the kraken project, available at https://github.com/mittagessen/kraken.
def nlbin_gpu(im: Image.Image, threshold: float = 0.5, estimate_zoom: float = 0.5, estimate_scale: float = 1.0, 
              border: float = 0.1, percentile: int = 80, percentile_range: int = 20, low: int = 5, 
              high: int = 90) -> tuple[Image.Image, Image.Image]:
    """
    Calculate a binary and a normalized image from a rgb image using GPU.
    Args:
        im: Input image.
        threshold: Binarization threshold.
        estimate_zoom: Zoom for page background estimation.
        estimate_scale: Scale for estimating a mask over the text region.
        border: Ignore this much of the border.
        percentile: Percentage for percentile filter.
        percentile_range: Range for percentile filter.
        low: Percentage for black estimation.
        high: Percentage for white estimation.
    Returns:
        A tuple containing the binary image and the normalized image.
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU modules not installed.")
    logger.debug("Convert image to grayscale")
    im = im.convert('L')
    logger.debug("Move image to GPU")
    raw = cp.array(im)

    logger.debug("Rescale image")
    raw = raw / float(cp.iinfo(raw.dtype).max)

    logger.debug("Normalize image")
    if cp.amax(raw) == cp.amin(raw):
        raise ValueError("Input image is empty")
    im = raw - cp.amin(raw)
    im /= cp.amax(im)
    with warnings.catch_warnings():  # hide all warnings:
        warnings.simplefilter("ignore", UserWarning)
        m= cp_zoom(im, estimate_zoom)
        m= cp_percentile_filter(m, percentile, size=(percentile_range, 2))
        m= cp_percentile_filter(m, percentile, size=(2, percentile_range))
        mh, mw = m.shape
        oh, ow = im.shape
        scale = cp.diag([mh * 1.0 / oh, mw * 1.0 / ow])
        m = cp_affine_transform(m, scale, output_shape=im.shape)  # rescale to original size
    w, h = cp.minimum(cp.array(im.shape), cp.array(m.shape))
    flat = cp.clip(im[:w, :h] - m[:w, :h] + 1, 0, 1)

    logger.debug("Estimate low and high thresholds")
    d0, d1 = flat.shape
    o0, o1 = int(border * d0), int(border * d1)
    est = flat[o0:d0 - o0, o1:d1 - o1]

    # by default, we use only regions that contain significant variance;
    # this makes the percentile based low and high estimates more reliable.
    v = est - cp_gaussian_filter(est, estimate_scale * 20.0)
    v = cp_gaussian_filter(v ** 2, estimate_scale * 20.0) ** 0.5
    v = (v > 0.3 * cp.amax(v))
    v = cp_binary_dilation(v, structure=cp.ones((int(estimate_scale * 50), 1)))
    v = cp_binary_dilation(v, structure=cp.ones((1, int(estimate_scale * 50))))
    est = est[v]
    lo = cp.percentile(est.ravel(), low)
    hi = cp.percentile(est.ravel(), high)
    flat -= lo
    flat /= (hi - lo)
    flat = cp.clip(flat, 0, 1)

    logger.debug("Create binary and normalized images")
    bin_im = cp.array(255 * (flat > threshold), 'B')
    bin_im = Image.frombytes("L", (bin_im.shape[1], bin_im.shape[0]), bin_im.tobytes())
    nrm_im = Image.fromarray(cp.asnumpy(flat * 255).astype(cp.uint8))
    return bin_im, nrm_im