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

import numpy as np
from PIL import Image
from scipy.ndimage import (
    affine_transform as np_affine_transform, 
    binary_dilation as np_binary_dilation,
    gaussian_filter as np_gaussian_filter,
    percentile_filter as np_percentile_filter,
    zoom as np_zoom
)

try:
    import cupy as cp
    from cupyx.scipy.ndimage import (
        affine_transform as cp_affine_transform,
        binary_dilation as cp_binary_dilation,
        gaussian_filter as cp_gaussian_filter,
        percentile_filter as cp_percentile_filter,
        zoom as cp_zoom
    )
    GPU_LIBRARY_AVAILABLE = True
except ModuleNotFoundError:
    GPU_LIBRARY_AVAILABLE = False


# This method is derived from nlbin Method (./kraken/binarization.py)
# in the kraken project, available at https://github.com/mittagessen/kraken.
def nlbin(
    im: Image.Image, 
    threshold: float = 0.5, 
    estimate_zoom: float = 0.5, 
    estimate_scale: float = 1.0, 
    border: float = 0.1, 
    percentile: int = 80, 
    percentile_range: int = 20, low: int = 5,
    high: int = 90,
    gpu: bool = False
) -> tuple[Image.Image, Image.Image]:
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
        A tuple containing the binary image [0] and the normalized image [1].
    """
    if gpu and not GPU_LIBRARY_AVAILABLE:
        raise RuntimeError("Could not find cupy")
    xp = cp if gpu else np
    affine_transform = cp_affine_transform if gpu else np_affine_transform
    binary_dilation = cp_binary_dilation if gpu else np_binary_dilation
    gaussian_filter = cp_gaussian_filter if gpu else np_gaussian_filter
    percentile_filter = cp_percentile_filter if gpu else np_percentile_filter
    zoom = cp_zoom if gpu else np_zoom
    
    im = im.convert('L')
    raw = xp.array(im)
    raw = raw / float(xp.iinfo(raw.dtype).max)
    
    # normalize image
    if xp.amax(raw) == xp.amin(raw):
        raise ValueError("Input image is empty")
    im = raw - xp.amin(raw)
    im /= xp.amax(im)
    with warnings.catch_warnings():  # hide all warnings:
        warnings.simplefilter("ignore", UserWarning)
        m= zoom(im, estimate_zoom)
        m= percentile_filter(m, percentile, size=(percentile_range, 2))
        m= percentile_filter(m, percentile, size=(2, percentile_range))
        mh, mw = m.shape
        oh, ow = im.shape
        scale = xp.diag([mh * 1.0 / oh, mw * 1.0 / ow])
        m = affine_transform(m, scale, output_shape=im.shape)  # rescale to original size
    w, h = xp.minimum(xp.array(im.shape), xp.array(m.shape))
    flat = xp.clip(im[:w, :h] - m[:w, :h] + 1, 0, 1)
    
    # estimate low and high thresholds
    d0, d1 = flat.shape
    o0, o1 = int(border * d0), int(border * d1)
    est = flat[o0:d0 - o0, o1:d1 - o1]
    
    # by default, we use only regions that contain significant variance;
    # this makes the percentile based low and high estimates more reliable.
    v = est - gaussian_filter(est, estimate_scale * 20.0)
    v = gaussian_filter(v ** 2, estimate_scale * 20.0) ** 0.5
    v = (v > 0.3 * xp.amax(v))
    v = binary_dilation(v, structure=xp.ones((int(estimate_scale * 50), 1)))
    v = binary_dilation(v, structure=xp.ones((1, int(estimate_scale * 50))))
    est = est[v]
    lo = xp.percentile(est.ravel(), low)
    hi = xp.percentile(est.ravel(), high)
    flat -= lo
    flat /= (hi - lo)
    flat = xp.clip(flat, 0, 1)
    
    # Convert arrays back to image
    bin_arr = xp.array(255 * (flat > threshold), 'B')
    nrm_arr = flat * 255
    if gpu:
        bin_arr = xp.asnumpy(bin_arr)
        nrm_arr = xp.asnumpy(nrm_arr)
    
    bin_im = Image.frombytes("L", (bin_arr.shape[1], bin_arr.shape[0]), bin_arr.tobytes())
    nrm_im = Image.fromarray(nrm_arr.astype(np.uint8))
    return bin_im, nrm_im
