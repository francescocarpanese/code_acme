# code-acme

**C**ontrol **O**rdinary **D**ifferential **E**quations using Deepmind [acme](https://github.com/deepmind/acme) framework. 

The purposes of code-acme package are: 
*   Implement simple physics environments, based on ode, to investigate continous action space control with deep reinforcement learning. 
*   Provide implementation examples of custom environments in [dm-control](https://github.com/deepmind/dm_control) framework.
*   Provide examples of using [acme](https://github.com/deepmind/acme) framework to train deep reinforcement learning agent for continuous action space control.

The implemented environments are meant to be lightwise to enable training with limited computational resources. The focus is to explore learning solutions and compare with standard linear control technique. 
However, by making use of [acme](https://github.com/deepmind/acme) framework, the experiment can be easily scaled up to allow for distributed learning on expensive environments and complex tasks. 

Ideally the project would serve as a tutorial for students and researchers to interfacing their own custom environments with [acme](https://github.com/deepmind/acme) frameworks and exploit DRL for control purposes. 

# Installation

# virtualenv
```
pip install virtualenv
virtualenv .code-acme
source .code-acme/bin/activate
pip install .[dev]
```

# Example/tutorials

# Future work 
