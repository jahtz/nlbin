# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from importlib.metadata import version
import logging
from pathlib import Path
from typing import Literal

import click
from PIL import Image
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn

from .nlbin import Nlbin, GPU_AVAILABLE
from .util import setup_logging, read_boolean_environment, expand_glob


logger: logging.Logger = logging.getLogger(__name__)
SHORT_HELP: bool = read_boolean_environment('NLBIN_EXTENDED_HELP', True)


@click.command(epilog='Developed at Centre for Philology and Digitality (ZPD), University of Würzburg')
@click.help_option('--help', hidden=SHORT_HELP)
@click.version_option(version('nlbin'), '--version', prog_name='nlbin', hidden=SHORT_HELP)
@click.argument(
    'images', 
    type=click.Path(), 
    callback=expand_glob, 
    nargs=-1, 
    required=True
)
@click.option(
    '-o', '--output',
    help='Output directory for generated image files. If omitted, each image file is written next to its input.',
    type=click.Path(file_okay=False, path_type=Path)
)
@click.option(
    '-d', '--device',
    help='Compute device for inference. Use "auto" to automatically choose the best available device.',
    type=click.Choice(['cpu', 'gpu', 'auto']),
    default='auto',
    show_default=True
)
@click.option(
    '-B', '--bin', 'binarize',
    help='Save the binary (black and white) version of each image.',
    is_flag=True
)
@click.option(
    '-N', '--nrm', 'normalize',
    help='Save the normalized (grayscale) version of each image.',
    is_flag=True
)
@click.option(
    '-b', '--bin-suffix',
    help='Specify the full suffix for binary images (including file extension).',
    default='.ocropus.bin.png',
    show_default=True
)
@click.option(
    '-n', '--nrm-suffix',
    help='Specify the full suffix for normalized images (including file extension).',
    default='.ocropus.nrm.png',
    show_default=True
)
@click.option(
    '--threshold',
    help='Binarization threshold. Higher values produce darker images.',
    type=click.FloatRange(0.0, 1.0), 
    default=0.5, 
    show_default=True,
    hidden=SHORT_HELP
)
@click.option(
    '--zoom', 'estimate_zoom',
    help='Zoom factor for background estimation. Smaller values focus on finer details.',
    type=click.FLOAT, 
    default=0.5, 
    show_default=True,
    hidden=SHORT_HELP
)
@click.option(
    '--scale', 'estimate_scale',
    help='Scale for text region mask estimation. Affects sensitivity to text features.',
    type=click.FLOAT, 
    default=1.0, 
    show_default=True,
    hidden=SHORT_HELP
)
@click.option(
    '--border',
    help='Fraction of image border to ignore to avoid edge artifacts.',
    type=click.FloatRange(0.0, 1.0),
    default=0.1, 
    show_default=True,
    hidden=SHORT_HELP
)
@click.option(
    '--percentile',
    help='Central percentile for background estimation.',
    type=click.IntRange(0, 100), 
    default=80, 
    show_default=True,
    hidden=SHORT_HELP
)
@click.option(
    '--range', 'percentile_range',
    help='Range around percentile for filtering.',
    type=click.IntRange(0, 100), 
    default=20, 
    show_default=True,
    hidden=SHORT_HELP
)
@click.option(
    '--low',
    help='Lower percentile for black pixel estimation.',
    type=click.IntRange(0, 100),
    default=5, 
    show_default=True,
    hidden=SHORT_HELP
)
@click.option(
    '--high',
    help='Upper percentile for white pixel estimation.',
    type=click.IntRange(0, 100), 
    default=90, 
    show_default=True,
    hidden=SHORT_HELP
)
@click.option(
     '--logging', 'level',
     help='Set logging level.', 
     type=click.Choice(['ERROR', 'WARNING', 'INFO', 'DEBUG']),
     default='ERROR',
     show_default=True
)
def cli(
    images: list[Path],
    output: Path | None = None,
    device: Literal['cpu', 'gpu', 'auto'] = 'auto',
    binarize: bool = False,
    normalize: bool = False,
    bin_suffix: str = '.ocropus.bin.png',
    nrm_suffix: str = '.ocropus.nrm.png',
    level: Literal['ERROR', 'WARNING', 'INFO', 'DEBUG'] = 'ERROR', 
    **kwargs
) -> None:
    """
    Normalize/Binarize RGB images using the OCRopus/nlbin algorithm.
    
    For GPU computation, a CuPy installation is requied. See README.md for further information.
    
    To view all available options with '--help', set the environment variable:
        NLBIN_EXTENDED_HELP=True
    
    IMAGES: One or more image paths. Use glob patterns in quotes to process multiple files.
    """
    setup_logging(level)
    
    if not images:
        raise click.BadArgumentUsage('No input images found')
    if not (binarize or normalize):
        raise click.BadOptionUsage('--bin', 'At least one of "-b/--binarize" or "-n/--normalize" has to bet set')
    if device == 'gpu' and not GPU_AVAILABLE:
        raise click.BadOptionUsage('--device', 'CUDA not available. Is a compatible version of CuPy installed?')
    if output is not None:
        output.mkdir(exist_ok=True, parents=True)
    
    with Progress(
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn('[progress.description]{task.description}'),
    ) as progress:
        nlbin: Nlbin = Nlbin(device)
        task = progress.add_task('Processing images', total=len(images))
        for fp in images:
            progress.update(task, description='/'.join(fp.parts[-4:]))
            logger.info(f'Processing image: {fp}')
            try:
                image: Image.Image = Image.open(fp)
                b, n = nlbin.process(image, **kwargs)
                out_dir: Path = output or fp.parent
                if binarize:
                    b_out_path: Path = out_dir / f'{fp.name.split(".")[0]}{bin_suffix}'
                    logger.debug(f'Writing binary output to {b_out_path}')
                    b.save(b_out_path)
                if normalize:
                    n_out_path: Path = out_dir / f'{fp.name.split(".")[0]}{nrm_suffix}'
                    logger.debug(f'Writing normalized output to {n_out_path}')
                    n.save(n_out_path)
            except Exception as exc:
                logger.error(f'Processing failed for {fp}: {exc}')
            progress.advance(task)
        progress.update(task, description='Done')
