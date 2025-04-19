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

import logging
from pathlib import Path
from typing import Optional, Union, Literal

from PIL import Image
import rich_click as click
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn

from nlbin import nlbin, nlbin_gpu, GPU_AVAILABLE
                           

__version__ = "0.2.0"
__prog__ = "nlbin"
__footer__ = "Developed at Centre for Philology and Digitality (ZPD), University of Würzburg"

DEVICES = Literal["cpu", "gpu"]

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(markup=True)]
)
logger = logging.getLogger("nlbin")


click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.MAX_WIDTH = 90
click.rich_click.RANGE_STRING = ""
click.rich_click.SHOW_METAVARS_COLUMN = False
click.rich_click.APPEND_METAVARS_HELP = True
click.rich_click.FOOTER_TEXT = __footer__
click.rich_click.OPTION_GROUPS = {
    "nlbin": [
        {
            "name": "Input",
            "options": ["images", "--glob", "--device"]
        },
        {
            "name": "Output",
            "options": ["--output", "--binarize", "--bin-suffix", "--normalize", "--nrm-suffix"]
        },
        {
            "name": "Fine-tuning",
            "options": ["--threshold", "--zoom", "--scale", "--border", "--percentage", "--range", "--low", "--high"]
        },
        {
            "name": "Help",
            "options": ["--help", "--version", "--verbose"],
        },
    ]
}
progress_bar = Progress(
    TextColumn("[progress.description]{task.description}"), 
    BarColumn(), 
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), 
    MofNCompleteColumn(), 
    TextColumn("•"), 
    TimeElapsedColumn(), 
    TextColumn("•"), 
    TimeRemainingColumn(), 
    TextColumn("• {task.fields[filename]}")
)

def callback_paths(ctx, param, value: Optional[list[str]]) -> list[Path]:
    """ Parse a list of click paths to a list of pathlib Path objects """
    return [] if value is None else list([Path(p) for p in value])

def callback_path(ctx, param, value: Optional[str]) -> Optional[Path]:
    """ Parse a click path to a pathlib Path object """
    return None if value is None else Path(value)

def callback_suffix(ctx, param, value: Optional[str]) -> Optional[str]:
    """ Parses a string to a valid suffix """
    return None if value is None else (value if value.startswith('.') else f".{value}")

def callback_logging(ctx, param, value: Optional[int]) -> int:
    """ Returns the logging level based on a verbosity counter (`0`: ERROR, `1`: WARNING, `2`: INFO, `>2`: DEBUG) """
    return 40 if value is None else 40 - (min(3, value) * 10)

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
                      message=f"{__prog__} v{__version__}\n{__footer__}")
@click.argument("images",
                type=click.Path(exists=True, dir_okay=True, file_okay=True, resolve_path=True),
                callback=callback_paths, required=True, nargs=-1)
@click.option("-g", "--glob", "glob",
              help="Glob pattern for matching images within directories. "
                   "Only applicable when directories are passed in IMAGES.",
              type=click.STRING, default="*.png", required=False, show_default=True)
@click.option("-o", "--output", "output",
              help="Specify output directory for processed files. Defaults to the parent directory of each input file.",
              type=click.Path(exists=False, dir_okay=True, file_okay=False, resolve_path=True),
              callback=callback_path, required=False)
@click.option("-B", "--binarize", "binarize",
              help="Save the binary (black and white) version of each image.", type=click.BOOL, is_flag=True)
@click.option("-N", "--normalize", "normalize",
              help="Save the normalized version of each image.", type=click.BOOL, is_flag=True)
@click.option("--bin-suffix", "bin_suffix",
              help="Specify suffix for binary output images.", type=click.STRING,
              callback=callback_suffix, required=False, default=".ocropus.bin.png", show_default=True)
@click.option("--nrm-suffix", "nrm_suffix",
              help="Specify suffix for normalized output images.", type=click.STRING,
              callback=callback_suffix, required=False, default=".ocropus.nrm.png", show_default=True)
@click.option("-d", "--device", "device",
              help=f"Select computation device. Use `gpu` for NVIDIA or AMD GPU acceleration (requires cupy). "
                   f"GPU available: {GPU_AVAILABLE}",
              type=click.Choice(["cpu", "gpu"]), default="cpu", required=False, show_default=True)
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
@click.option("-v", "--verbose", "verbosity",
              help="Set verbosity level. `-v`: WARNING, `-vv`: INFO, `-vvv`: DEBUG.", 
              type=click.INT, count=True, callback=callback_logging)
def cli(images: list[Path], glob: str = "*.png", output: Optional[Path] = None, binarize: bool = False, 
        normalize: bool = False, bin_suffix: str = ".ocropus.bin.png", nrm_suffix: str = ".ocropus.nrm.png", 
        device: DEVICES = "cpu", threshold: float = 0.5, estimate_zoom: float = 0.5, estimate_scale: float = 0.5, 
        border: float = 0.1, percentile: int = 80, percentile_range: int = 20, low: int = 5, high: int = 90, 
        verbosity: int = 40):
    """
    Normalize and binarize images using OCRopus nlbin algorithm.

    IMAGES: List of image file paths to process. Accepts individual files,
    wildcards, or directories (with -g option for pattern matching).
    
    If you want to use your GPU, consider installing cupy. See README.md for further information.
    """
    logger.setLevel(verbosity)
    
    if not (binarize or normalize):
        raise click.BadOptionUsage("--binarize", "Either --binarize or --normalize has to bet set.")
    
    if device == "gpu" and not GPU_AVAILABLE:
        raise click.BadOptionUsage("--device", "CUDA not available. Is a compatible version of cupy installed?")
    
    logger.info("Loading images")
    images = expand_paths(images, glob)
    if not images:
        raise click.BadArgumentUsage("No images found.")
    logger.info(f"{len(images)} images found")
    
    if output is not None:
        output.mkdir(exist_ok=True, parents=True)
    
    with progress_bar as bar:
        task = bar.add_task("Processing images", total=len(images), filename="")
        for fp in images:
            bar.update(task, filename=Path("/", *fp.parts[-min(len(fp.parts), 4):]))
            try:
                image = Image.open(fp)
                base_name = fp.name.split('.')[0]
                base_dir = fp.parent if output is None else output
                if device == "cpu":
                    bin_im, nrm_im = nlbin(image, threshold=threshold, estimate_zoom=estimate_zoom,
                                           estimate_scale=estimate_scale, border=border, percentile=percentile,
                                           percentile_range=percentile_range, low=low, high=high)
                elif device == "gpu":
                    bin_im, nrm_im = nlbin_gpu(image, threshold=threshold, estimate_zoom=estimate_zoom,
                                               estimate_scale=estimate_scale, border=border, percentile=percentile,
                                               percentile_range=percentile_range, low=low, high=high)
                if binarize:
                    bin_im.save(base_dir.joinpath(base_name + bin_suffix))
                if normalize:
                    nrm_im.save(base_dir.joinpath(base_name + nrm_suffix))
            except Exception as e:
                logger.error(f"Processing failed for file {fp.as_posix()}: {e}")
            bar.update(task, advance=1)
        bar.update(task, filename="Done")


if __name__ == "__main__":
    cli()
