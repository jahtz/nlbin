# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import glob
from os import getenv
from pathlib import Path

import click


def read_boolean_environment(name: str, invert: bool = False) -> bool:
    """ Read a boolean value from an env variable """
    v: str | None = getenv(name)
    if v is None or v.strip().lower() not in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True if invert else False
    else:
        return False if invert else True
    
    
def expand_glob(ctx: click.Context, param: click.Parameter, patterns: list[str]) -> list[Path]:
    """ Expand glob expressions in path strings """
    paths: list[Path] = []
    for pattern in patterns:
        if glob.has_magic(pattern):
            for match in glob.iglob(pattern, recursive=True):
                path: Path = Path(match)
                if path.is_file():
                    paths.append(path.resolve())
        else:
            path: Path = Path(pattern)
            if path.is_file() and path.exists():
                paths.append(path.resolve())
    return paths
