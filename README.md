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

  To view all available options with '--help', set the environment variable:
  NLBIN_EXTENDED_HELP=True

  IMAGES: One or more image paths. Use glob patterns in quotes to process
  multiple files.

Options:
  -o, --output DIRECTORY          Output directory for generated image files.
                                  If omitted, each image file is written next
                                  to its input.
  -d, --device [cpu|gpu|auto]     Compute device for inference. Use "auto" to
                                  automatically choose the best available
                                  device.  [default: auto]
  -B, --bin                       Save the binary (black and white) version of
                                  each image.
  -N, --nrm                       Save the normalized (grayscale) version of
                                  each image.
  -b, --bin-suffix TEXT           Specify the full suffix for binary images
                                  (including file extension).  [default:
                                  .ocropus.bin.png]
  -n, --nrm-suffix TEXT           Specify the full suffix for normalized
                                  images (including file extension).
                                  [default: .ocropus.nrm.png]
  --logging [ERROR|WARNING|INFO|DEBUG]
                                  Set logging level.  [default: ERROR]

  Developed at Centre for Philology and Digitality (ZPD), University of
  Würzburg
```

## ZPD

Developed at Centre for [Philology and Digitality](https://www.uni-wuerzburg.de/en/zpd/) (ZPD), [University of Würzburg](https://www.uni-wuerzburg.de/en/).