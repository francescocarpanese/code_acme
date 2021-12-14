import matplotlib.pyplot as plt
import numpy as np
from environments.tank_dm_control import Tasks
from environments.tank_dm_control import tank
from dm_control.rl.control import Environment

# Test default init
def test_default_init():
    env = Environment(tank.physics(),Tasks.HoldTarget())
    TimeStep = env.reset()
    assert all(np.isfinite(TimeStep.observation)), 'Standard initialization-> observations not finite'
    assert all(np.isfinite(env._physics._init_state)), 'Initial state not finite'

# Test time stepping on default init with constant actions
def test_N_timesteps():

    # Instantiate env
    environment = Environment(tank.physics(),Tasks.HoldTarget())
    TimeStep = environment.reset()

    # Define null constant actions
    actions = np.array([0.], dtype=np.float32)
    
    # Store observations and rewards
    N_t = 30
    cumulative_reward = 0.
    o = np.zeros([2, N_t])
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

    plt.figure()
    plt.plot(t[0,:],np.transpose(o))
    plt.xlabel('t [s]')
    plt.ylabel('x')

    # TODO Deal with this with verbosity
    plt.figure()
    plt.plot(t[0,:],o[1, :])
    plt.xlabel('t [s]')
    plt.ylabel('h')

    plt.figure()
    plt.plot(t[0,:],r[0, :])
    plt.xlabel('t [s]')
    plt.ylabel('r')

def test_time_stepping_time_limit():
    # Instantiate env
    environment = Environment(tank.physics(),Tasks.HoldTarget(), time_limit= 2.)
    TimeStep = environment.reset()

    # Define null constant actions
    actions = np.array([0.], dtype=np.float32)
    
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

    plt.figure()
    plt.plot(t,o)
    plt.xlabel('t [s]')
    plt.ylabel('x')

    plt.figure()
    plt.plot(t,r)
    plt.xlabel('t [s]')
    plt.ylabel('r')

    # TODO Deal with this with verbosity
    plt.figure()
    plt.plot(t,[x[1] for x in o])
    plt.xlabel('t [s]')
    plt.ylabel('h')



# Test tend 
def test_tend():
   # TODO
   pass

# Test termination criterion 
def test_termination():
    # TODO
    pass

# Test bahavior with not finite actions
def test_behavior_to_not_normal_actions():
    # TODO
    pass

# Test different dt for simulation and action:
def test_dt_sim_vs_dt_ctr():
    # TODO
    pass

# Run from main for debugging
if __name__ == "__main__":
    test_default_init()
    test_N_timesteps()
    test_termination()
    test_tend()
    test_time_stepping_time_limit()