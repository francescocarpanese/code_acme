# python3
"""Install script for setuptools."""

# TODO clean up and split requirements for testing, dev etc

from setuptools import find_packages
from setuptools import setup

# Core requirements for installation
install_requirements = []

# Specifics for the tagged version 
# https://github.com/deepmind/acme/releases/tag/0.2.2
# The zip file contains the requirements for the version.

# Define extra requirements depending on user case
dm_requirements = [
    'dm-acme==0.2.2',
    'dm-acme[jax,tf,launchpad,envs]==0.2.2',
    'tfp-nightly==0.14.0.dev20210818', # Force the same version as in released and not from pypi
    'tensorflow_probability==0.14.1', # dm-acme v0.2.2 comes with
]

test_requirements = [
    'pytest-xdist',
    'pylint'
]

dev_requirements = [
    'pytest-xdist',
    'ipykernel',
    'ipython',
] + dm_requirements + test_requirements

# Freeze requirements as in dm_acme==0.2.2 release for longstanding colab tutorials
colab_requirements = [
    'absl-py==0.12.0',
    'atari-py==0.2.9',
    'bsuite==0.3.5',
    'dataclasses==0.8',
    'dm-control==0.0.364896371',
    'dm-env==1.5',
    'dm-haiku==0.0.4',
    'dm-launchpad-nightly==0.3.0.dev20210818',
    'dm-reverb==0.4.0',
    'dm-sonnet==2.0.0',
    'dm-tree==0.1.6',
    'jax==0.2.17',
    'jaxlib==0.1.68',
    'jax==0.2.19',
    'jaxlib==0.1.70',
    'keras==2.6.0',
    'optax==0.0.9',
    'Pillow==8.3.1',
    'pytype==2021.8.11',
    'pytest-xdist==2.3.0',
    'rlax==0.0.4',
    'tensorflow-datasets==4.4.0',
    'tensorflow-estimator==2.6.0',
    'tensorflow==2.6.0',
    'tfp-nightly==0.14.0.dev20210818',
    'trfl==1.2.0',
]

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
        'colab': colab_requirements,
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
