# Copyright (c) Facebook, Inc. and its affiliates. MIT license.
"""Build the inherited PointNet++ extension without depending on cwd."""
from pathlib import Path
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sources = sorted(str(p.relative_to(ROOT)) for p in (ROOT / '_ext_src/src').glob('*.cpp'))
sources += sorted(str(p.relative_to(ROOT)) for p in (ROOT / '_ext_src/src').glob('*.cu'))
setup(name='motiontrend-pointnet2', version='0.1.0', packages=['pointnet2'],
      ext_modules=[CUDAExtension('pointnet2._ext', sources=sources,
          include_dirs=[str(ROOT / '_ext_src/include')],
          extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']})],
      cmdclass={'build_ext': BuildExtension})
