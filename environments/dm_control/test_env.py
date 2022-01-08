import matplotlib.pyplot as plt
import numpy as np
from environments.dm_control import tank
from dm_control.rl.control import Environment
import pytest
from environments.dm_control import MovingCoil0D,tank

@pytest.fixture(scope="module", params=["Moving_Coil", "tank"])
def get_env(request):
    if request.param == "tank":
        yield tank
    elif request.param == "Moving_Coil":
        yield MovingCoil0D

def test_get_parameter_dict(get_env):
    physics = get_env.Physics.physics()
    assert any(physics.get_par_dict()), 'Empty parameter dictionary'

def test_default_init(get_env):
    env = Environment(get_env.Physics.physics(),get_env.Tasks.Step())
    TimeStep = env.reset()
    assert all(np.isfinite(TimeStep.observation)), 'Standard initialization-> observations not finite'
    assert all(np.isfinite(env._physics._init_state)), 'Initial state not finite'

def test_N_timesteps(get_env):
    # Instantiate env
    environment = Environment(get_env.Physics.physics(),get_env.Tasks.Step())
    TimeStep = environment.reset()
    
    # Define null constant actions
    actions = np.zeros(environment.action_spec().shape, np.float32)

    # Store observations and rewards
    N_t = 5
    cumulative_reward = 0.
    o = np.zeros([environment.observation_spec().shape[0], N_t])
    r = np.zeros([1,N_t])
    t = np.zeros([1,N_t])
    for it in range(N_t):
        TimeStep = environment.step(actions)
        o[:, it] = TimeStep.observation
        r[:, it] = TimeStep.reward
        t[:, it] = environment.physics.time()
        cumulative_reward += TimeStep.reward

    message = 'Stepping env with default init and constant action provides not finite '
    assert np.isfinite(o).all(), message + 'observations'
    assert np.isfinite(r).all(), message + 'rewards'

def test_time_stepping_time_limit(get_env):
    # Instantiate env
    environment = Environment(get_env.Physics.physics(),get_env.Tasks.Step(), time_limit= 0.5)
    TimeStep = environment.reset()

    # Define null constant actions
    actions = np.zeros(environment.action_spec().shape, np.float32)
    
    # Store observations and rewards
    cumulative_reward = 0.
    o, r, t = [],[],[]
    while not TimeStep.last():
        TimeStep = environment.step(actions)
        o.append(TimeStep.observation.tolist())
        r.append(TimeStep.reward)
        t.append(environment.physics.time())
        cumulative_reward += TimeStep.reward

    message = 'Stepping env with default init and constant action provides not finite '
    assert np.isfinite(o).all(), message + 'observations'
    assert np.isfinite(r).all(), message + 'rewards'