from environments.MovingCoil0D.Moving_Coil import Moving_Coil
import matplotlib.pyplot as plt
import numpy as np

# Test default init
def test_default_init():
    environment = Moving_Coil()
    TimeStep = environment.reset()
    assert all(np.isfinite(TimeStep.observation)), 'Standard initialization-> observations not finite'
    assert np.isfinite(environment._x_goal), 'Goal definition not finite'
    assert all(np.isfinite(environment._init_state)), 'Initial state not finite'

# Test time stepping on default init with constant actions
def test_time_stepping():

    # Instantiate env
    environment = Moving_Coil()
    TimeStep = environment.reset()

    # Define constant actions
    actions = np.array([1., 2.], dtype=np.float32)
    
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

    # TODO Deal with this with verbosity
    plt.figure()
    plt.plot(t[0,:],o[0, :])
    plt.xlabel('t [s]')
    plt.ylabel('x')

    plt.figure()
    plt.plot(t[0,:],o[1, :])
    plt.xlabel('t [s]')
    plt.ylabel('v')

    plt.figure()
    plt.plot(t[0,:],r[0, :])
    plt.xlabel('t [s]')
    plt.ylabel('r')

# Test tend 
def test_tend():
    # Define constant actions to trigger instability
    actions = np.array([1., 2.], dtype=np.float32)
    # The following parameters are chosen such that 
    # the physical divergence termination is not reached 
    tend = 0.5
    dt_sim = 1e-1
    dt_ctr = 2*dt_sim
    
    env = Moving_Coil(tend=tend, dt_sim=dt_sim, dt_ctr=dt_ctr)
    TimeStep = env.reset()
    it = 0
    while not TimeStep.last():
        TimeStep = env.step(actions)
        it += 1
    assert it == int(tend/dt_ctr) + 1, 'Physics solver terminated with wrong number of steps'
    assert env._t >= tend, 'Pysics solver terminated prematurely'

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