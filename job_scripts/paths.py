"""Resolve VSC directory aliases on the login host using only Python stdlib.

Keep bin/python as a symlink so a venv does not become its base interpreter.
"""
import argparse
from pathlib import Path


def directory(value):
    return str(Path(value).expanduser().resolve())


def executable(value):
    path=Path(value).expanduser().absolute()
    return str(path.parent.resolve()/path.name)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('path');p.add_argument('--executable',action='store_true')
    a=p.parse_args()
    print(executable(a.path) if a.executable else directory(a.path))
