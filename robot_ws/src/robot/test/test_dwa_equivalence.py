import numpy as np
from pathlib import Path
from robot.automatic import Automatic
from robot import robot_config as cfg

# Golden file created by benchmarks/dwa_bench.py --save-reference (old DWA)
REFERENCE_PATH = Path(__file__).parent / "data" / "dwa_reference.npz"


def load_reference():
    # We load the golden file
    
    golden = np.load(REFERENCE_PATH)
    return golden

def current_dwa(vel, goal, obs):
    # The only place that knows HOW to call the DWA
    # (if the DWA moves to dwa.py later, only this function changes)

    # FCreating the node
    node = Automatic.__new__(Automatic)
    # Setting up velocity and goal
    node.vx_cur, node.vy_cur, node.wz_cur = vel
    gx, gy = goal

    # Running the DWA
    cmd, score = node._dwa(gx, gy, obs)
    return cmd, score


def test_config_unchanged():
    # The golden file is only valid if the DWA's config is the same as when it was made
    ref = load_reference()

    for key, saved in zip(ref["config_keys"], ref["config_values"]):
        current = float(getattr(cfg, str(key)))
        assert current == saved, f"{key} changed: {saved} -> {current}, golden file no longer valid"


def test_same_decisions():
    ref = load_reference()
    failures = []

    for i in range(len(ref["obs_len"])):
        # Rebuild the inputs of case i
        start = ref["obs_start"][i]
        length = ref["obs_len"][i]
        obs = ref["obs"][start:start + length]
        goal = ref["goal"][i]
        vel = ref["vel"][i]

        # Current DWA
        new_cmd, new_score = current_dwa(vel, goal, obs)

        # Old DWA decisions
        ref_cmd = ref["cmd"][i]
        ref_score = ref["score"][i]

        # Case 1: the old DWA found no valid command (saved as NaN)
        if np.isnan(ref_cmd[0]):
            if new_cmd is not None:
                failures.append(f"case {i}: expected no command, got {new_cmd}")
            continue

        # Case 2: the old DWA found a command, the new one must find one too
        if new_cmd is None:
            failures.append(f"case {i}: expected {ref_cmd}, got no command")
            continue

        # The score must be the same
        if not np.isclose(new_score, ref_score, rtol=0, atol=1e-9):
            failures.append(f"case {i}: score {new_score} instead of {ref_score}")
            continue

        # Same score but a different command is a tie: acceptable
        # (two candidates with the same score, both are correct)

    assert not failures, f"{len(failures)} cases differ:\n" + "\n".join(failures[:10])