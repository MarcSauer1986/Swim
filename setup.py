#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io

from os.path import dirname
from os.path import join

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    file_path = join(dirname(__file__), *names)
    with io.open(file_path, encoding=kwargs.get('encoding', 'utf8')) as fh:
        return fh.read()


setup(
    name='Swim',
    version='0.1.0',
    license='MIT license',
    description='Neue Fische - Digitales Gesellenstück',
    long_description=read('README.md'),
    author='Marc Sauer',
    author_email='marcsauer@web.de',
    url='https://github.com/MarcSauer1986/Swim',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    zip_safe=False,
)
