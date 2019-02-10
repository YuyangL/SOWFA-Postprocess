from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
        ext_modules = cythonize("PostProcess_EnergySpectrum.pyx"),
        # ext_modules = cythonize("PostProcess_AnisotropyTensor.pyx"),
        include_dirs=[numpy.get_include()]
        )
