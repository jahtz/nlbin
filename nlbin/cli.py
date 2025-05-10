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

import glob
from pathlib import Path
from typing import Optional

import click
from PIL import Image
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn

from nlbin.nlbin import nlbin, GPU_LIBRARY_AVAILABLE


__version__ = "0.2.2"
__prog__ = "nlbin"
__footer__ = "Developed at Centre for Philology and Digitality (ZPD), University of Würzburg"

PROGRESS = Progress(
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

def callback_paths(ctx, param, value) -> list[Path]:
    if not value:
        raise click.BadParameter("", param=param)
    paths = []
    for pattern in value:
        expanded = glob.glob(pattern, recursive=True)
        if not expanded:
            p = Path(pattern)
            if p.exists() and p.is_file():
                paths.append(p)
        else:
            paths.extend(Path(p) for p in expanded if Path(p).is_file())
    if not paths:
        raise click.BadParameter("None of the provided paths or patterns matched existing files.")
    return paths


@click.command(epilog=__footer__)
@click.help_option("--help")
@click.version_option(
    __version__, "--version",
    prog_name=__prog__,
    message=f"{__prog__} v{__version__}\n{__footer__}"
)
@click.argument(
    "images",
    type=click.Path(exists=False, dir_okay=True, file_okay=True, resolve_path=True),
    required=True,
    callback=callback_paths,
    nargs=-1
)
@click.option(
    "-o", "--output", "output",
    help="Specify output directory for processed files. Defaults to the parent directory of each input file.",
    type=click.Path(exists=False, dir_okay=True, file_okay=False, resolve_path=True, path_type=Path)
)
@click.option(
    "-b", "--binarize", "binarize",
    help="Save the binary (black and white) version of each image.", 
    type=click.BOOL, 
    is_flag=True
)
@click.option(
    "-n", "--normalize", "normalize",
    help="Save the normalized version of each image.", 
    type=click.BOOL, 
    is_flag=True
)
@click.option(
    "--bin-suffix", "bin_suffix",
    help="Specify suffix for binary output images.", 
    type=click.STRING,
    default=".ocropus.bin.png", 
    show_default=True
)
@click.option(
    "--nrm-suffix", "nrm_suffix",
    help="Specify suffix for normalized output images.", 
    type=click.STRING,
    default=".ocropus.nrm.png", 
    show_default=True
)
@click.option(
    "--gpu/--cpu", "gpu",
    help=f"Select computation device. Use '--gpu' for CUDA or ROCm GPU acceleration (requires cupy). "
         f"GPU available: {GPU_LIBRARY_AVAILABLE}",
    type=click.BOOL,
    default=GPU_LIBRARY_AVAILABLE,
    show_default=True
)
@click.option(
    "--threshold", "threshold",
    help="Set binarization threshold.",
    type=click.FloatRange(0.0, 1.0), 
    default=0.5, 
    show_default=True
)
@click.option(
    "--zoom", "estimate_zoom",
    help="Zoom level for estimating page background.",
    type=click.FloatRange(0.0, 1.0), 
    default=0.5, 
    show_default=True
)
@click.option(
    "--scale", "estimate_scale",
    help="Scale factor for defining the text region mask.",
    type=click.FloatRange(0.0, 1.0), 
    default=1.0, 
    show_default=True
)
@click.option(
    "--border", "border",
    help="Fraction of the image border to ignore in processing.",
    type=click.FloatRange(0.0, 1.0),
    default=0.1, 
    show_default=True
)
@click.option(
    "--percentage", "percentile",
    help="Percentile value for image filtering to enhance contrast.",
    type=click.IntRange(0, 100), 
    default=80, 
    show_default=True
)
@click.option(
    "--range", "percentile_range",
    help="Range for the percentile filter to adjust brightness/contrast.",
    type=click.IntRange(0, 100), 
    default=20, 
    show_default=True
)
@click.option(
    "--low", "low",
    help="Lower percentage threshold for black level estimation.",
    type=click.IntRange(0, 100),
    default=5, 
    show_default=True
)
@click.option(
    "--high", "high",
    help="Upper percentage threshold for white level estimation.",
    type=click.IntRange(0, 100), 
    default=90, 
    show_default=True
)
def cli(
    images: list[Path],
    output: Optional[Path] = None,
    binarize: bool = False, 
    normalize: bool = False, 
    bin_suffix: str = ".ocropus.bin.png", 
    nrm_suffix: str = ".ocropus.nrm.png",
    gpu: bool = GPU_LIBRARY_AVAILABLE,
    threshold: float = 0.5, 
    estimate_zoom: float = 0.5, 
    estimate_scale: float = 0.5, 
    border: float = 0.1, 
    percentile: int = 80, 
    percentile_range: int = 20, 
    low: int = 5, 
    high: int = 90, 
) -> None:
    """
    Normalize and binarize images using OCRopus nlbin algorithm.

    IMAGES: List of image file paths to process. Accepts individual files, glob wildcards, or directories.
    
    If you want to use your GPU, consider installing cupy. See README.md for further information.
    """
    if not images:
        raise click.BadArgumentUsage("No images found")
    if not (binarize or normalize):
        raise click.BadOptionUsage("--binarize", "Either '-b/--binarize' or '-n/--normalize' has to bet set")
    if gpu and not GPU_LIBRARY_AVAILABLE:
        raise click.BadOptionUsage("--gpu", "CUDA not available. Is a compatible version of cupy installed?")
    if output is not None:
        output.mkdir(exist_ok=True, parents=True)
    bin_suffix = bin_suffix if bin_suffix.startswith('.') else '.' + bin_suffix
    nrm_suffix = nrm_suffix if nrm_suffix.startswith('.') else '.' + nrm_suffix
    
    with PROGRESS as progressbar:
        task = progressbar.add_task("Processing...", total=len(images), filename="")
        for fp in images:
            progressbar.update(task, filename=Path(*fp.parts[-min(len(fp.parts), 4):]))
            try:
                img = Image.open(fp)
                fn = fp.name.split('.')[0]
                outd = output if output else fp.parent
                bin_im, nrm_im = nlbin(
                    img,
                    gpu=gpu,
                    threshold=threshold,
                    estimate_zoom=estimate_zoom,
                    estimate_scale=estimate_scale, 
                    border=border, 
                    percentile=percentile,
                    percentile_range=percentile_range, 
                    low=low, 
                    high=high
                )
                if binarize:
                    bin_im.save(outd.joinpath(fn + bin_suffix))
                if normalize:
                    nrm_im.save(outd.joinpath(fn + nrm_suffix))
            except Exception as e:
                progressbar.log(f"Processing failed for file {fp.as_posix()}:\n{e}")
            progressbar.advance(task)
        progressbar.update(task, filename="Done")
