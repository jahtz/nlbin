# nlbin
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

Implements the OCRopus/nlbin algorithm with CPU and GPU (CUDA) support.
Uses adaptive thresholding and percentile-based background estimation for robust document image binarization.

## Credits
Based on the algorithm introduced by Afzal et al. in:<br>
_Robust binarization of stereo and monocular document images using percentile filter_

Forked from:
- [OCRopus nlbin](https://github.com/ocropus-archive/DUP-ocropy/blob/master/ocropus-nlbin)
- [Kraken](https://github.com/mittagessen/kraken)

## Setup

> [!NOTE]
> The setup process is configured for [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/jahtz/nlbin && cd nlbin
```

**CPU-only**
```bash
uv tool install .
```

**GPU (CUDA) support**
```bash
uv tool install .[cuda12]     # For CUDA 12.x
uv tool install .[cuda12ctk]  # For CUDA without Toolkit (requires only a CUDA driver)
uv tool install .[cuda13]     # For CUDA 13.x
uv tool install .[cuda13ctk]  # For CUDA without Toolkit (requires only a CUDA driver)
```

## Usage
```bash
nlbin [OPTIONS] IMAGES...
```

```text
$ nlbin --help
Usage: nlbin [OPTIONS] IMAGES...

  Normalize/Binarize RGB images using the OCRopus/nlbin algorithm.

  For GPU computation, a CuPy installation is requied. See README.md for
  further information.

  Arguments:
    IMAGES: One or more image paths. Use glob patterns in quotes to process 
            multiple files.

Options:
  -o, --output DIRECTORY       Output directory for generated image files. If
                               omitted, each image file is written next to its
                               input.
  -d, --device [cpu|gpu|auto]  Compute device for inference. Use "auto" to
                               automatically choose the best available device.
                               [default: auto]
  -B, --binarize               Save the binary (black and white) version of
                               each image.
  -N, --normalize              Save the normalized (grayscale) version of each
                               image.
  -b, --bin-suffix TEXT        Specify the full suffix for binary images
                               (including file extension).  [default:
                               .ocropus.bin.png]
  -n, --nrm-suffix TEXT        Specify the full suffix for normalized images
                               (including file extension).  [default:
                               .ocropus.nrm.png]
  --threshold FLOAT RANGE      Binarization threshold. Higher values produce
                               darker images.  [default: 0.5; 0.0<=x<=1.0]
  --zoom FLOAT                 Zoom factor for background estimation. Smaller
                               values focus on finer details.  [default: 0.5]
  --scale FLOAT                Scale for text region mask estimation. Affects
                               sensitivity to text features.  [default: 1.0]
  --border FLOAT RANGE         Fraction of image border to ignore to avoid
                               edge artifacts.  [default: 0.1; 0.0<=x<=1.0]
  --percentile INTEGER RANGE   Central percentile for background estimation.
                               [default: 80; 0<=x<=100]
  --range INTEGER RANGE        Range around percentile for filtering.
                               [default: 20; 0<=x<=100]
  --low INTEGER RANGE          Lower percentile for black pixel estimation.
                               [default: 5; 0<=x<=100]
  --high INTEGER RANGE         Upper percentile for white pixel estimation.
                               [default: 90; 0<=x<=100]
  -v, --verbose                Set the verbosity level. Default: ERROR. Use -v
                               for WARNING, -vv for INFO, -vvv for DEBUG.

  Developed at Centre for Philology and Digitality (ZPD), University of
  Würzburg
```

## ZPD

Developed at Centre for [Philology and Digitality](https://www.uni-wuerzburg.de/en/zpd/) (ZPD), [University of Würzburg](https://www.uni-wuerzburg.de/en/).