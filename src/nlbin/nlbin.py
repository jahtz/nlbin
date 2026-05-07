# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from typing import Literal
import warnings

from PIL import Image
import numpy as np
import scipy.ndimage as np_ndimage
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    GPU_AVAILABLE: bool = True
except (ModuleNotFoundError, ImportError):
    GPU_AVAILABLE: bool = False


logger: logging.Logger = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = int(10e9)


class Nlbin:
    """
    Implements the OCRopus/nlbin algorithm with CPU and GPU (CUDA) support.
    Uses adaptive thresholding and percentile-based background estimation for robust document image binarization.
    """
    
    def __init__(self, device: Literal['cpu', 'gpu', 'auto'] = 'auto') -> None:
        self.device = device.lower()
        
        if self.device == 'auto':
            self.device = 'gpu' if GPU_AVAILABLE else 'cpu'
        
        if self.device == 'gpu' and not GPU_AVAILABLE:
            raise RuntimeError('GPU requested but not available.')
        
        if self.device == 'gpu':
            self.xp = cp
            self.ndimage = cp_ndimage
        elif self.device == 'cpu':
            self.xp = np
            self.ndimage = np_ndimage
        else:
            raise ValueError(f'Invalid device: {device}')
        
        logger.info(f'Computation backend: {self.device.upper()}')
    
    def process(
        self,
        image: Image.Image,
        threshold: float = 0.5, 
        estimate_zoom: float = 0.5, 
        estimate_scale: float = 1.0, 
        border: float = 0.1, 
        percentile: int = 80, 
        percentile_range: int = 20, 
        low: int = 5,
        high: int = 90
    ) -> tuple[Image.Image, Image.Image]:
        """
        Generate binary and normalized grayscale images from an RGB input.
        Args:
            image: Input RGB image.
            threshold: Binarization threshold. Higher values produce darker images. Defaults to 0.5.
            estimate_zoom: Zoom factor for background estimation. Smaller values focus on finer details. 
                Defaults to 0.5.
            estimate_scale: Scale for text region mask estimation. Affects sensitivity to text features. 
                Defaults to 1.0.
            border: Fraction of image border to ignore to avoid edge artifacts. Defaults to 0.1.
            percentile: Central percentile for background estimation. Defaults to 80.
            percentile_range: Range around percentile for filtering. Defaults to 20.
            low: Lower percentile for black pixel estimation. Defaults to 5
            high: Upper percentile for white pixel estimation. Defaults to 90
        Returns:
            Tuple of (binary image, normalized grayscale image).
        """
        logger.debug('Load image')
        image = image.convert('L')
        raw = self.xp.array(image)
        raw = raw / float(self.xp.iinfo(raw.dtype).max)
        
        logger.debug('Normalize image')
        if self.xp.amax(raw) == self.xp.amin(raw):
            raise ValueError('Input image is empty')
        image = raw - self.xp.amin(raw)
        image /= self.xp.amax(image)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            m = self.ndimage.zoom(image, estimate_zoom)
            m= self.ndimage.percentile_filter(m, percentile, size=(percentile_range, 2))
            m= self.ndimage.percentile_filter(m, percentile, size=(2, percentile_range))
            mh, mw = m.shape
            oh, ow = image.shape
            scale = self.xp.diag([mh * 1.0 / oh, mw * 1.0 / ow])
            m = self.ndimage.affine_transform(m, scale, output_shape=image.shape)  # rescale to original size
        w, h = self.xp.minimum(self.xp.array(image.shape), self.xp.array(m.shape))
        flat = self.xp.clip(image[:w, :h] - m[:w, :h] + 1, 0, 1)
        
        logger.debug('Estimate low and high thresholds')
        d0, d1 = flat.shape
        o0, o1 = int(border * d0), int(border * d1)
        est = flat[o0:d0 - o0, o1:d1 - o1]
        
        logger.debug('Flatten image')
        v = est - self.ndimage.gaussian_filter(est, estimate_scale * 20.0)
        v = self.ndimage.gaussian_filter(v ** 2, estimate_scale * 20.0) ** 0.5
        v = (v > 0.3 * self.xp.amax(v))
        v = self.ndimage.binary_dilation(v, structure=self.xp.ones((int(estimate_scale * 50), 1)))
        v = self.ndimage.binary_dilation(v, structure=self.xp.ones((1, int(estimate_scale * 50))))
        est = est[v]
        lo = self.xp.percentile(est.ravel(), low)
        hi = self.xp.percentile(est.ravel(), high)
        flat -= lo
        flat /= (hi - lo)
        flat = self.xp.clip(flat, 0, 1)
        
        logger.debug('Convert arrays back to image objects')
        bin_arr = self.xp.array(255 * (flat > threshold), 'B')
        nrm_arr = flat * 255
        
        if self.device == 'gpu':
            bin_arr = self.xp.asnumpy(bin_arr)  # ty:ignore[unresolved-attribute]
            nrm_arr = self.xp.asnumpy(nrm_arr)  # ty:ignore[unresolved-attribute]
        
        return (
            Image.frombytes('L', (bin_arr.shape[1], bin_arr.shape[0]), bin_arr.tobytes()),
            Image.fromarray(nrm_arr.astype(np.uint8))
        )
