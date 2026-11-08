"""Load only the point operators included beside this source package."""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def bind():
    # Relative imports cannot silently select a globally installed pointnet2.
    from . import pointnet2_utils
    return pointnet2_utils


def extension_path():
    return str(Path(bind()._ext.__file__).resolve())
