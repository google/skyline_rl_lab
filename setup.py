from __future__ import print_function
from setuptools import setup
import io
import codecs
import os
import sys
import skyline


here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    # intentionally *not* adding an encoding option to open
    return codecs.open(os.path.join(here, *parts), 'r').read()

long_description = read('README.md')

setup(
    name='skyline-rl-lab',
    version=skyline.__version__,
    description='A package to provide RL capability in tasks.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/google/skyline_rl_lab',
    author='John Lee',
    author_email='puremonkey2007@gmail.com',
    license='MIT License',
    packages=[
        'skyline', 'skyline.alg', 'skyline.lab',
    ],
    install_requires=[
        'contourpy==1.2.1',
        'cycler==0.12.1',
        'fonttools==4.53.1',
        'immutabledict==4.2.0',
        'kiwisolver==1.4.5',
        'matplotlib==3.9.2',
        'numpy==2.0.1',
        'packaging==24.1',
        'pillow==10.4.0',
        'prettytable==3.11.0',
        'pyparsing==3.1.2',
        'python-dateutil==2.9.0.post0',
        'six==1.16.0',
        'tqdm==4.66.5',
        'wcwidth==0.2.13',
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
