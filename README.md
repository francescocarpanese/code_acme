<p align="center">
  <img src="docs/images/code_acme_(1).jpg" width="50%">
</p>

# code_acme

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
[![test](https://github.com/francescocarpanese/code_acme/actions/workflows/ci.yaml/badge.svg)](https://github.com/francescocarpanese/code_acme/actions/workflows/ci.yaml)


**C**ontrol **O**rdinary **D**ifferential **E**quations using Deepmind [acme](https://github.com/deepmind/acme) framework. 

code-acme goals: 
*   Implement simple physics environments, based on ode, to investigate continous action space control with deep reinforcement learning. 
*   Provide implementation examples of custom environments with [dm-control](https://github.com/deepmind/dm_control) framework specifics.
*   Provide examples of using [acme](https://github.com/deepmind/acme) framework to train deep reinforcement learning agent for continuous action space control.

The implemented environments are meant to be lightwise to enable training with limited computational resources. 
However, thanks to [acme](https://github.com/deepmind/acme) framework, the experiments can be easily scaled up to allow for distributed learning.

Ideally the project would serve as a tutorial for students and researchers to interfacing their own custom environments with [acme](https://github.com/deepmind/acme) frameworks and exploit deep reinforcement learning for continuous control purposes. 

# Installation
Checkout the git repository.

## docker container
We recommand docker installation. 

Build docker image.

```
make build 
```

Run bash on docker image. 
```
make bash
```

The docker image includes all the package dependencies for training including `tensorflow`, `acme`,`dm_control`.
Running `make bash` will mount `code-acme` folder as a [docker volume](https://docs.docker.com/storage/bind-mounts/). 
This allows you to develop your code within the docker container or outside with your favourite environment. 

From within docker bash, install `code-acme` package to make sure you are using the latest version of the package, including eventually your modifications. 
```
pip install .
```

To test the installation.
```
pytest
```

## virtualenv
! 

```
pip install virtualenv
virtualenv .code-acme
source .code-acme/bin/activate
pip install .[dev]
```

## Test installation 
```
pytest -v -m "not slow"
```

# Example/tutorials

# Future plans 
Below some high priority features I plan to incorporate: 
- Examples of training agent with distributed learning using [launchpad](https://github.com/deepmind/launchpad)
- Improve flexibility in storing checkpoints and snapshots during training following solution in [mava](https://github.com/instadeepai/Mava)
- Add utils for hyperparameter scan and tuning with [wandb](https://wandb.ai/site)
- Improve parameters handling with [hydra](https://hydra.cc/docs/intro/)
- Implement more complex tasks and compare drl performances with linear control solution.

# Contributing
If you have any question reach out at `francesco [dot] carpanese [at] hotmail [dot] it` or open an new issue. 
