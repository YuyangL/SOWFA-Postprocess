from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

system = 'windows'  # 'windows', 'unix'
fileName = 'PostProcess_EnergySpectrum'
# fileName = 'PostProcess_AnisotropyTensor'

if system == 'unix':
    ext_modules = [Extension(fileName,
                            [fileName + '.pyx'],
                            libraries=["m"],  # Unix-like specific
                            extra_compile_args = ['-ffast-math', '-O3'])]
else:
    ext_modules = [Extension(fileName,
                             [fileName + '.pyx'])]

setup(name = fileName,
      cmdclass = {'build_ext': build_ext},
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
