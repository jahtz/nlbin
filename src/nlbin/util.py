# SPDX-License-Identifier: Apache-2.0
import glob
import logging
from pathlib import Path

import click
from rich.logging import RichHandler


def setup_logging(level: int = 0) -> None:
    logging.basicConfig(
        level=max(10, 40 - (10 * level)),
        format='%(message)s', 
        datefmt='[%X]', 
        handlers=[RichHandler(markup=True, rich_tracebacks=True)]
    )
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.info(f'Logging verbosity set to {logging.getLevelName(logger.getEffectiveLevel())}')
    
    
def expand_glob(ctx: click.Context, param: click.Parameter, patterns: list[str] | str) -> list[Path]:
    if isinstance(patterns, str):
        patterns = [patterns]
    paths: list[Path] = []
    for pattern in patterns:
        if glob.has_magic(pattern):
            for match in glob.iglob(pattern, recursive=True):
                p = Path(match)
                if p.is_file():
                    paths.append(p.resolve())
        else:
            p = Path(pattern)
            if p.is_file() and p.exists():
                paths.append(p.resolve())
    return paths
