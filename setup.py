#MIT License

#Copyright (c) 2022 Francesco Carpanese

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

"""Install script for setuptools."""

from setuptools import find_packages
from setuptools import setup


# Core requirements for installation
install_requirements = [
    'dm-acme[tf,jax,testing,envs]==0.3.0',
    'dm-control',
]

# Define extra requirements depending on user case
test_requirements = [
    'pytest-xdist',
    'pylint'
    ]

dev_requirements = [
    'ipykernel',
    'ipython',
    ] + test_requirements

long_description = """code-acme is a library wich provides light physics
 environments based on ordinary differential equation, wraps them as a 
 https://github.com/deepmind/dm_control 
 environements and makes use of 
 https://github.com/deepmind/acme
 framework to train agents for continuous action space control.
 It aims to provides examples to researchers that would like to 
 approach deep reinforcement learning as a technique for control purposed, 
 exploring solutions without prohibitive computational costs. 
 
 For more information see [git repository]https://github.com/francescocarpanese/code_acme"""

setup(
    name='code-acme',
    version='0.0',
    description='Examples DRL to control simple ODE',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Francesco Carpanese',
    license='TODO',
    keywords='deep reinforcement-learning python machine learning control ODE',
    packages=find_packages(),
    install_requires=install_requirements,
    extras_require={
        'dev':  dev_requirements,
        'test': test_requirements,
    },
    url="https://github.com/cisk1990/code-acme",
    classifiers=[
        'Development Status :: 0.0',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Control Engineering',
        'Topic :: Scientific/Engineering :: Deep Reinforcement Learning',
    ],
)
