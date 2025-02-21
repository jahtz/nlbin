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

from pathlib import Path
from typing import Optional, Union, Literal

import rich_click as click
from PIL import Image
from rich import print as rprint
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn, TimeElapsedColumn

from nlbin import nlbin, nlbin_gpu

try:
    import cupy as cp
    cupy_available = True
except ModuleNotFoundError:
    cupy_available = False


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
