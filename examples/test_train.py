import subprocess
import pytest

def capture(command):
    proc = subprocess.Popen(command,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
    )
    out,err = proc.communicate()
    return out, err, proc.returncode

@pytest.mark.parametrize("environment", ["tank","moving_coil"])
@pytest.mark.parametrize("agent", ["mpo","dmpo","d4pg"])
@pytest.mark.slow
def test_1episode_training(agent, environment):
    """
    Test 1 training episode for combination of environment/agent
    """
    command = ["python", "examples/train.py",
               "--num_episodes", "1",
               "--agent",agent,
               "--environment", environment]
    _, _, exitcode = capture(command)
    assert exitcode == 0
