<p align="center">
  <img src="docs/images/code_acme_(1).jpg" width="50%">
</p>

# code_acme

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
[![test](https://github.com/francescocarpanese/code_acme/actions/workflows/ci.yaml/badge.svg)](https://github.com/francescocarpanese/code_acme/actions/workflows/ci.yaml)


**C**ontrol **O**rdinary **D**ifferential **E**quations using Deepmind [acme](https://github.com/deepmind/acme) framework. 

The purposes of code-acme package are: 
*   Implement simple physics environments, based on ode, to investigate continous action space control with deep reinforcement learning. 
*   Provide implementation examples of custom environments in [dm-control](https://github.com/deepmind/dm_control) framework.
*   Provide examples of using [acme](https://github.com/deepmind/acme) framework to train deep reinforcement learning agent for continuous action space control.

The implemented environments are meant to be lightwise to enable training with limited computational resources. The focus is to explore learning solutions and compare with standard linear control technique. 
However, by making use of [acme](https://github.com/deepmind/acme) framework, the experiment can be easily scaled up to allow for distributed learning on expensive environments and complex tasks. 

Ideally the project would serve as a tutorial for students and researchers to interfacing their own custom environments with [acme](https://github.com/deepmind/acme) frameworks and exploit deep reinforcement learning for control purposes. 

# Installation

## docker container

```
make build 
```

```
make bash
pip install .
```



## virtualenv
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

# Future work 
