#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    pept is a Python library that unifies Positron Emission Particle
#    Tracking (PEPT) research, including tracking, simulation, data analysis
#    and visualisation tools
#
#    Copyright (C) 2019-2021 the pept developers
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

# File   : setup.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 23.08.2019

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine


import  io
import  os
import  sys
import  warnings
from    shutil              import  rmtree

from    setuptools          import  find_packages, setup, Command, Extension


try:
    import  numpy               as      np
    from    Cython.Build        import  cythonize
    from    Cython.Distutils    import  build_ext
except ImportError as e:
    warnings.warn(e.args[0])
    warnings.warn((
        'The pept package requires Cython and Numpy to be pre-installed'
    ))
    raise ImportError((
        'Cython or Numpy not found! Please install Cython and Numpy (or run '
        '`pip install -r requirements.txt`) and try again.'
    ))


# Package meta-data.
NAME = 'pept'
DESCRIPTION = (
    'A Python library that unifies Positron Emission Particle Tracking (PEPT) '
    'research, including tracking, simulation, data analysis and '
    'visualisation tools.'
)
URL = 'https://github.com/uob-positron-imaging-centre/pept'
EMAIL = 'a.l.nicusan@bham.ac.uk'
AUTHOR = 'Andrei Leonard Nicusan'
REQUIRES_PYTHON = '>=3.7.0'


def requirements(filename):
    # The dependencies are the same as the contents of requirements.txt
    with open(filename) as f:
        return [line.strip() for line in f if line.strip()]


# What packages are required for this module to be executed?
REQUIRED = requirements('requirements.txt')


# What packages are optional?
EXTRAS = {
    'docs': requirements('requirements_extra.txt'),
}

# Cythonize keyword arguments
cythonize_kw = dict(language_level = 3)

# Compiler arguments for each extension - with *full* optimisations.
# Unix-specific compiler args are followed by MSVC ones; they will be filtered
# based on the compiler used in `BuildExtCompilerSpecific`
cy_extension_kw = dict()
cy_extension_kw['extra_compile_args'] = [
    '-Ofast', '-flto', '/O2', '/fp:fast', '/GL'
]
cy_extension_kw['extra_link_args'] = ['-flto', '/LTCG']
cy_extension_kw['include_dirs'] = [np.get_include()]

# Compiler arguments for each extension - with *strict floating point*
# optimisations.
cy_extension_kw_strict = dict()
cy_extension_kw_strict['extra_compile_args'] = [
    '-O3', '-flto', '/O2', '/GL'
]
cy_extension_kw_strict['extra_link_args'] = ['-flto', '/LTCG']
cy_extension_kw_strict['include_dirs'] = [np.get_include()]

cy_extensions = [
    Extension(
        'pept.scanners.parallel_screens.extensions.binary_converter',
        ['pept/scanners/parallel_screens/extensions/binary_converter.pyx'],
        **cy_extension_kw_strict
    ),
    Extension(
        'pept.utilities.cutpoints.find_cutpoints',
        ['pept/utilities/cutpoints/find_cutpoints.pyx'],
        **cy_extension_kw
    ),
    Extension(
        'pept.utilities.cutpoints.find_minpoints',
        ['pept/utilities/cutpoints/find_minpoints.pyx'],
        **cy_extension_kw
    ),
    Extension(
        'pept.tracking.tof.cutpoints_tof',
        ['pept/tracking/tof/cutpoints_tof.pyx'],
        **cy_extension_kw
    ),
    Extension(
        'pept.scanners.modular_camera.extensions.get_pept_event',
        ['pept/scanners/modular_camera/extensions/get_pept_event.pyx'],
        **cy_extension_kw
    ),
    Extension(
        'pept.utilities.traverse.traverse3d',
        ['pept/utilities/traverse/traverse3d.pyx'],
        **cy_extension_kw_strict
    ),
    Extension(
        'pept.utilities.traverse.traverse2d',
        ['pept/utilities/traverse/traverse2d.pyx'],
        **cy_extension_kw_strict
    ),
    Extension(
        'pept.tracking.trajectory_separation.distance_matrix_reachable',
        ['pept/tracking/trajectory_separation/distance_matrix_reachable.pyx'],
        **cy_extension_kw
    ),
    Extension(
        'pept.tracking.birmingham_method.extensions.birmingham_method',
        ['pept/tracking/birmingham_method/extensions/birmingham_method.pyx'],
        **cy_extension_kw
    ),
    Extension(
        'pept.tracking.fpi.fpi_ext',
        ['pept/tracking/fpi/fpi_ext.pyx'],
        **cy_extension_kw
    ),
]

extensions = cythonize(cy_extensions, **cythonize_kw)


class BuildExtCompilerSpecific(build_ext):
    '''Before compiling extensions, ensure only valid compiler arguments are
    used - e.g. MSVC expects "/O2", while GCC and Clang expect "-O3".
    '''
    def build_extensions(self):
        # If compiling under MSVC, only allow "/*" compiler arguments
        if "msvc" in self.compiler.compiler_type.lower():
            for ext in self.extensions:
                ext.extra_compile_args = [
                    ca for ca in ext.extra_compile_args if ca.startswith("/")
                ]

                ext.extra_link_args = [
                    la for la in ext.extra_link_args if la.startswith("/")
                ]

        # Otherwise only allow compiler arguments starting with "-"
        else:
            for ext in self.extensions:
                ext.extra_compile_args = [
                    ca for ca in ext.extra_compile_args if ca.startswith("-")
                ]

                ext.extra_link_args = [
                    la for la in ext.extra_link_args if la.startswith("-")
                ]

        build_ext.build_extensions(self)


here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# Load the package's __version__.py module as a dictionary.
about = {}
project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
with open(os.path.join(here, project_slug, '__version__.py')) as f:
    exec(f.read(), about)


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


# Where the magic happens:
setup(
    name = NAME,
    version = about['__version__'],
    description = DESCRIPTION,
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    author = AUTHOR,
    author_email = EMAIL,
    python_requires = REQUIRES_PYTHON,
    url = URL,
    packages = find_packages(
        exclude = ["tests", "*.tests", "*.tests.*", "tests.*"]
    ),
    install_requires = REQUIRED,
    extras_require = EXTRAS,
    include_package_data = True,
    keywords = 'pept positron emission particle tracking',
    license = 'GNU',
    classifiers = [
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Cython',
        'Programming Language :: C',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    # $ setup.py publish support.
    cmdclass = {
        'upload': UploadCommand,
        'build_ext': BuildExtCompilerSpecific,
    },
    ext_modules = extensions,
)
