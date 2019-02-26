from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules=[ Extension("PostProcess_AnisotropyTensor",
                        ["PostProcess_AnisotropyTensor.pyx"],
                        extra_compile_args = ["-ffast-math", "-O3"])]

setup(
      name = "PostProcess_AnisotropyTensor",
      cmdclass = {"build_ext": build_ext},
      ext_modules = ext_modules,
      include_dirs = [numpy.get_include()])

# from distutils.core import setup
# from Cython.Build import cythonize
# import numpy
#
# setup(
#         # ext_modules = cythonize("PostProcess_EnergySpectrum.pyx"),
#         ext_modules = cythonize("PostProcess_AnisotropyTensor.pyx"),
#         include_dirs=[numpy.get_include()]
#         )
