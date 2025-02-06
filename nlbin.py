# Copyright 2024 Janik Haitz
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
from pathlib import Path
from typing import Optional, Union, Literal

import rich_click as click
import numpy as np
from PIL import Image
from rich import print as rprint
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn, TimeElapsedColumn
from scipy.ndimage import affine_transform, percentile_filter, gaussian_filter, binary_dilation, zoom

try:
    import cupy as cp
    from cupyx.scipy.ndimage import affine_transform as cp_affine_transform
    from cupyx.scipy.ndimage import percentile_filter as cp_percentile_filter
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter
    from cupyx.scipy.ndimage import binary_dilation as cp_binary_dilation
    from cupyx.scipy.ndimage import zoom as cp_zoom
    cupy_available = True
except ModuleNotFoundError:
    cupy_available = False


# Config
__version__ = "1.2"
__prog__ = "nlbin"
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.MAX_WIDTH = 90
click.rich_click.RANGE_STRING = ""
click.rich_click.OPTION_GROUPS = {
    "*": [
        {
            "name": "Input",
            "options": ["images"]
        },
        {
            "name": "Options",
            "options": ["--binary", "--normalized", "--glob", "--output", "--device",
                        "--keep-suffixes", "--bin-suffix", "--nrm-suffix"]
        },
        {
            "name": "Fine-tuning",
            "options": ["--threshold", "--zoom", "--scale", "--border", "--percentage", "--range", "--low", "--high"]
        },
        {
            "name": "Help",
            "options": ["--help", "--version"]
        }
    ],
}
progress = Progress(TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    MofNCompleteColumn(),
                    TextColumn("•"),
                    TimeElapsedColumn(),
                    TextColumn("•"),
                    TimeRemainingColumn(),
                    TextColumn("• {task.fields[filename]}"))
DEVICES = Literal["cpu", "gpu"]

# Logic
# This method is derived from nlbin Method (./kraken/binarization.py)
# in the kraken project, available at https://github.com/mittagessen/kraken.
def nlbin(im: Image,
          threshold: float = 0.5,
          estimate_zoom: float = 0.5,
          estimate_scale: float = 1.0,
          border: float = 0.1,
          percentile: int = 80,
          percentile_range: int = 20,
          low: int = 5,
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
    # convert to grayscale first
    im = im.convert('L')
    raw = np.array(im)

    # rescale image
    raw = raw / float(np.iinfo(raw.dtype).max)

    # normalize image
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

    # estimate low and high thresholds
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

    # create binary and normalized images
    bin_im = np.array(255 * (flat > threshold), 'B')
    bin_im = Image.frombytes("L", (bin_im.shape[1], bin_im.shape[0]), bin_im.tobytes())
    nrm_im = Image.fromarray((flat * 255).astype(np.uint8))
    return bin_im, nrm_im


# This method is derived from nlbin Method (./kraken/binarization.py)
# in the kraken project, available at https://github.com/mittagessen/kraken.
def nlbin_gpu(im: Image,
              threshold: float = 0.5,
              estimate_zoom: float = 0.5,
              estimate_scale: float = 1.0,
              border: float = 0.1,
              percentile: int = 80,
              percentile_range: int = 20,
              low: int = 5,
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
    im = im.convert('L')
    raw = cp.array(im.convert('L'))

    # rescale image
    raw = raw / float(cp.iinfo(raw.dtype).max)

    # normalize image
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

    # estimate low and high thresholds
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

    # create binary and normalized images
    bin_im = cp.array(255 * (flat > threshold), 'B')
    bin_im = Image.frombytes("L", (bin_im.shape[1], bin_im.shape[0]), bin_im.tobytes())
    nrm_im = Image.fromarray(cp.asnumpy(flat * 255).astype(cp.uint8))
    return bin_im, nrm_im


# CLI
def paths_callback(ctx, param, value: list[str]) -> Optional[list[Path]]:
    """ Parse a list of click paths to a list of pathlib Path objects """
    return [] if value is None else list([Path(p) for p in value])


def path_callback(ctx, param, value: str) -> Optional[Path]:
    """ Parse a click path to a pathlib Path object """
    return None if value is None else Path(value)


def suffix_callback(ctx, param, value: str) -> str:
    """ Parses a string to a valid suffix """
    return value if value.startswith('.') else f".{value}"


def expand_paths(paths: Union[Path, list[Path]], glob: str = '*') -> list[Path]:
    """Expands a list of paths by unpacking directories."""
    result = []
    if isinstance(paths, list):
        for path in paths:
            if path.is_dir():
                result.extend([p for p in path.glob(glob) if p.is_file()])
            else:
                result.append(path)
    elif isinstance(paths, Path):
        if paths.is_dir():
            result.extend([p for p in paths.glob(glob) if p.is_file()])
        else:
            result.append(paths)
    return sorted(result)


@click.command()
@click.help_option("--help")
@click.version_option(__version__,
                      "--version",
                      prog_name=__prog__,
                      message=f"{__prog__} v{__version__} - Developed at Centre for Philology and Digitality (ZPD), "
                              f"University of Würzburg")
@click.argument("images",
                type=click.Path(exists=True, dir_okay=True, file_okay=True, resolve_path=True),
                callback=paths_callback, required=True, nargs=-1)
@click.option("-B", "--binary", "binary",
              help="Save the binary (black and white) version of each image.", type=click.BOOL, is_flag=True)
@click.option("-N", "--normalized", "normalized",
              help="Save the normalized version of each image.", type=click.BOOL, is_flag=True)
@click.option("-g", "--glob", "glob",
              help="Glob pattern for matching images within directories. "
                   "Only applicable when directories are passed in IMAGES.",
              type=click.STRING,
              default="*.png", required=False, show_default=True)
@click.option("-o", "--output", "output",
              help="Specify output directory for processed files. Defaults to the parent directory of each input file.",
              type=click.Path(exists=False, dir_okay=True, file_okay=False, resolve_path=True),
              callback=path_callback, required=False)
@click.option("-d", "--device", "device",
              help="Select computation device. Use `gpu` for NVIDIA or AMD GPU acceleration (requires cupy).",
              type=click.Choice(["cpu", "gpu"]),
              default="cpu", required=False, show_default=True)
@click.option("-s", "--keep-suffixes", "keep_suffixes",
              help="Preserve all filename suffixes except the last one. If not set, removes all suffixes.",
              type=click.BOOL, is_flag=True)
@click.option("--bin-suffix", "bin_suffix",
              help="Specify suffix for binary output images.", type=click.STRING,
              callback=suffix_callback, required=False, default=".ocropus.bin.png", show_default=True)
@click.option("--nrm-suffix", "nrm_suffix",
              help="Specify suffix for normalized output images.", type=click.STRING,
              callback=suffix_callback, required=False, default=".ocropus.nrm.png", show_default=True)
@click.option("--threshold", "threshold",
             help="Set binarization threshold.",
             type=click.FloatRange(0.0, 1.0), required=False, default=0.5, show_default=True)
@click.option("--zoom", "estimate_zoom",
              help="Zoom level for estimating page background.",
              type=click.FloatRange(0.0, 1.0), required=False, default=0.5, show_default=True)
@click.option("--scale", "estimate_scale",
              help="Scale factor for defining the text region mask.",
              type=click.FloatRange(0.0, 1.0), required=False, default=1.0, show_default=True)
@click.option("--border", "border",
              help="Fraction of the image border to ignore in processing.",
              type=click.FloatRange(0.0, 1.0), required=False, default=0.1, show_default=True)
@click.option("--percentage", "percentile",
              help="Percentile value for image filtering to enhance contrast.",
              type=click.IntRange(0, 100), required=False, default=80, show_default=True)
@click.option("--range", "percentile_range",
              help="Range for the percentile filter to adjust brightness/contrast.",
              type=click.IntRange(0, 100), required=False, default=20, show_default=True)
@click.option("--low", "low",
              help="Lower percentage threshold for black level estimation.",
              type=click.IntRange(0, 100), required=False, default=5, show_default=True)
@click.option("--high", "high",
              help="Upper percentage threshold for white level estimation.",
              type=click.IntRange(0, 100), required=False, default=90, show_default=True)
def cli(images: list[Path], output: Optional[Path] = None, glob: str = "*.png", device: DEVICES = "cpu",
        binary: bool = False, normalized: bool = False, keep_suffixes: bool = False,
        bin_suffix: str = ".ocropus.bin.png", nrm_suffix: str = ".ocropus.nrm.png", threshold: float = 0.5,
        estimate_zoom: float = 0.5, estimate_scale: float = 0.5, border: float = 0.1, percentile: int = 80,
        percentile_range: int = 20, low: int = 5, high: int = 90):
    """
    Generate binary and normalized versions of input images.

    IMAGES: List of image file paths to process. Accepts individual files,
    wildcards, or directories (with -g option for pattern matching).
    """
    images = expand_paths(images, glob)
    if not images:
        rprint("[bold red]Error:[/bold red] No images found!")
        return
    if not (binary or normalized):
        rprint("[bold red]Error:[/bold red] Either [green]--bin[/green] or [green]--nrm[/green] has to be set.")
        return
    if device == "gpu" and not cupy_available:
        rprint("[bold red]Error:[/bold red] CUDA not available. "
               "Is a compatible version of [green]cupy[/green] installed?")
    rprint(f"{len(images)} images found")
    if output is not None:
        output.mkdir(exist_ok=True, parents=True)

    with progress as p:
        task = p.add_task("Generating images...", total=len(images), filename="")
        for fp in images:
            p.update(task, filename=fp)
            try:
                image = Image.open(fp)
                out_dir = fp.parent if output is None else output
                base = fp.stem if keep_suffixes else fp.name.split('.')[0]
                if device == "cpu":
                    bin_im, nrm_im = nlbin(image, threshold=threshold, estimate_zoom=estimate_zoom,
                                           estimate_scale=estimate_scale, border=border, percentile=percentile,
                                           percentile_range=percentile_range, low=low, high=high)
                elif device == "gpu":
                    bin_im, nrm_im = nlbin_gpu(image, threshold=threshold, estimate_zoom=estimate_zoom,
                                               estimate_scale=estimate_scale, border=border, percentile=percentile,
                                               percentile_range=percentile_range, low=low, high=high)
                else:
                    rprint(f"[bold red]Error:[/bold red] Invalid device: {device}")
                    return
                if binary:
                    bin_im.save(out_dir.joinpath(base + bin_suffix))
                if normalized:
                    nrm_im.save(out_dir.joinpath(base + nrm_suffix))
            except Exception as e:
                p.log(f"Processing failed for file {fp.as_posix()} ({e})")
            p.update(task, advance=1)
        p.update(task, filename="Done!")


if __name__ == "__main__":
    cli()
