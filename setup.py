#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.


import codecs
import os
from setuptools import setup, find_packages

from carpedm import __version__


here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the relevant file
with codecs.open(os.path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='carpedm',
      version=__version__,
      description='Character shapes image metadata manager for machine learning.',
      long_description=long_description,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
      ],
      keywords='machine learning framework deep learning image-to-text japanese character',
      url='http://github.com/SimulatedANeal/carpedm',
      author='Neal Digre',
      author_email='nealdigre@ed-alumni.net',
      license='MIT',
      packages=find_packages(),
      python_requires='~=3.4',
      install_requires=[
          'numpy>=1.14',
          'tensorflow>=1.5',
          'Pillow>=5.1',
      ],
      extras_require={
          'plot': [
              'matplotlib>=2.1.2',
          ],
          'tf': [
              'tensorflow>=1.5',
          ],
          'tfgpu': [
              'tensorflow-gpu>1.5',
          ],
          'docs': [
              'sphinx >= 1.7',
              'sphinx_rtd_theme',
          ]
      },
      tests_require=[
          'nose',
          'matplotlib>=2.1.2',
      ],
      test_suite='nose.collector',
      entry_points={
          'console_scripts': [
              'download_data = carpedm.data.download:main',
          ],
      },
      include_package_data=True,
      zip_safe=False)
