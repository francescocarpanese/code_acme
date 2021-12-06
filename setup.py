# python3
"""Install script for setuptools."""

# TODO clean up and split requirements for testing, dev etc

from setuptools import find_packages
from setuptools import setup

# Core requirements for installation
install_requirements = []

# Specifics for the tagged version https://github.com/deepmind/acme/releases/tag/0.2.2
# The zip file contains the requirements for the version.

# Define extra requirements depending on user case
dm_requirements = [
    'dm-acme==0.2.2',
    'dm-acme[jax,tf,launchpad,envs]==0.2.2',
    'tfp-nightly==0.14.0.dev20210818', # Force the same version as in released and not in pypi
    'tensorflow_probability==0.14.1', # dm-acme v0.2.2 comes with
]

test_requirements = [
    'pytest-xdist',
]

dev_requirements = [
    'pytest-xdist',
    'ipykernel',
    'ipython',
] + dm_requirements + test_requirements

# Extras
long_description = """TODO  """

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
        'dm':  dm_requirements,
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