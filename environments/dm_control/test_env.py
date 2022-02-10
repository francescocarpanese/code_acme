"""Test environment: physics and task """
import numpy as np
from dm_control.rl.control import Environment
import pytest
from environments.dm_control import moving_coil,tank,moving_coil2D

@pytest.fixture(scope="module", params=["Moving_Coil", "tank", "Moving_Coil2D"])
def get_env(request):
    """Returns environment modules"""
    if request.param == "tank":
        yield tank
    elif request.param == "Moving_Coil":
        yield moving_coil
    elif request.param == "Moving_Coil2D":
        yield moving_coil2D

def test_get_parameter_dict(get_env): # pylint: disable=missing-function-docstring,redefined-outer-name
    physics = get_env.physics.Physics()
    assert any(physics.get_par_dict()), 'Empty parameter dictionary'

def test_default_init(get_env): # pylint: disable=missing-function-docstring,redefined-outer-name
    env = Environment(get_env.physics.Physics(),get_env.tasks.Step())
    timestep = env.reset()
    assert all(np.isfinite(timestep.observation)),\
         'Standard initialization-> observations not finite'
    assert all(np.isfinite(env._physics.get_state())),\
         'Initial state not finite' # pylint: disable=protected-access

def test_n_timesteps(get_env): # pylint: disable=redefined-outer-name
    """Tests n time steps of environment with 0 action"""
    # Instantiate env
    environment = Environment(get_env.physics.Physics(),get_env.tasks.Step())
    timestep = environment.reset()

    # Define null constant actions
    actions = np.zeros(environment.action_spec().shape, np.float32)

    # Store observations and rewards
    n_t = 5
    cumulative_reward = 0.
    obs = np.zeros([environment.observation_spec().shape[0], n_t])
    rew = np.zeros([1,n_t])
    time = np.zeros([1,n_t])
    for it in range(n_t):
        timestep = environment.step(actions)
        obs[:, it] = timestep.observation
        rew[:, it] = timestep.reward
        time[:, it] = environment.physics.time()
        cumulative_reward += timestep.reward

    message = """
        Stepping env with default init and constant action
        provides not finite"""
    assert np.isfinite(obs).all(), message + 'observations'
    assert np.isfinite(rew).all(), message + 'rewards'

def test_time_stepping_time_limit(get_env): # pylint: disable=redefined-outer-name
    """Tests stepping env till specified time limit"""
    # Instantiate env
    environment = Environment(
        get_env.physics.Physics(),
        get_env.tasks.Step(),
        time_limit= 0.5,
    )

    timestep = environment.reset()

    # Define null constant actions
    actions = np.zeros(environment.action_spec().shape, np.float32)

    # Store results
    cumulative_reward = 0.
    obs, rew, time = [],[],[]
    while not timestep.last():
        timestep = environment.step(actions)
        obs.append(timestep.observation.tolist())
        rew.append(timestep.reward)
        time.append(environment.physics.time())
        cumulative_reward += timestep.reward

    message = """
        Stepping env with default init and
        constant action provides not finite"""
    assert np.isfinite(obs).all(), message + 'observations'
    assert np.isfinite(rew).all(), message + 'rewards'
