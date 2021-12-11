import matplotlib.pyplot as plt
import numpy as np
from environments.tank import Tasks
from environments.tank.tank import env

# Test default init
def test_default_init():
    environment = env()
    TimeStep = environment.reset()
    assert all(np.isfinite(TimeStep.observation)), 'Standard initialization-> observations not finite'
    assert all(np.isfinite(environment._init_state)), 'Initial state not finite'

# Test time stepping on default init with constant actions
def test_time_stepping():

    # Instantiate env
    environment = env()
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
        t[:, it] = environment._t
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
    test_time_stepping()
    test_termination()
    test_tend()