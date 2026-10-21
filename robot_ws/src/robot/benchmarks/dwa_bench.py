import argparse
import subprocess
import time
from pathlib import Path

import numpy as np

from robot.automatic import Automatic
from robot import robot_config as cfg

# Default location of the golden file: robot_ws/src/robot/test/data/dwa_reference.npz
DEFAULT_REFERENCE_PATH = Path(__file__).resolve().parents[1] / "test" / "data" / "dwa_reference.npz"

# Config values the DWA's decisions depend on saved in the golden file 
# so we know they haven't changed when we rerun the benchmark
CONFIG_KEYS = [
    "AUTO_LOOP_HZ",
    "DWA_MAX_VX", "DWA_MAX_VY", "DWA_MAX_WZ", "DWA_MAX_ACC_V", "DWA_MAX_ACC_W",
    "DWA_V_SAMPLES", "DWA_W_SAMPLES", "DWA_SIM_TIME", "DWA_SIM_STEPS",
    "DWA_ROBOT_RADIUS", "DWA_OBSTACLE_MARGIN", "DWA_LETHAL_DIST",
    "DWA_W_HEADING", "DWA_W_CLEARANCE", "DWA_W_VELOCITY", "DWA_W_GOAL_DIST",
    "LIDAR_MAX_RANGE_M",
]


def wall_points(p0, p1, n):
    # We create an array of points that will be our wall
    t = np.linspace(0.0, 1.0, n)
    wall = p0 + t[:, None] * (p1 - p0)
    return wall


def random_wall(r_range=(0.4, 1.8), len_range=(0.5, 2.0)):
    # Centre of the wall, in polar coordinates
    r = np.random.uniform(*r_range)
    a = np.random.uniform(-np.pi, np.pi)
    c = r * np.array([np.cos(a), np.sin(a)])

    # Orientation and length of the wall
    phi = np.random.uniform(-np.pi, np.pi)
    d = np.array([np.cos(phi), np.sin(phi)])
    L = np.random.uniform(*len_range)

    # Calculation of the wall's endpoints
    p0 = c - (L / 2) * d
    p1 = c + (L / 2) * d
    return p0, p1


def make_scene(M, n_walls=4, min_dist=0.25, max_dist=cfg.LIDAR_OBSTACLE_RANGE,
               max_attempts=100):
    # Create some extra points cause filtering will remove some
    pts_per_wall = int(1.05 * M) // n_walls

    for _ in range(max_attempts):
        # One array of points per wall, then stacked into a single (~M, 2) array
        walls = [wall_points(*random_wall(), pts_per_wall) for _ in range(n_walls)]
        pts = np.vstack(walls)

        # We keep the points beyween min_dist and max_dist from the robot
        dist = np.hypot(pts[:, 0], pts[:, 1])
        pts = pts[(dist > min_dist) & (dist < max_dist)]

        if len(pts) >= M:
            break
    else:
        raise RuntimeError(f"Could not generate {M} points in {max_attempts} attempts")

    obs = pts

    # Goal: random direction, 0.5 to 3 m away
    g_r = np.random.uniform(0.5, 3.0)
    g_a = np.random.uniform(-np.pi, np.pi)
    goal = (g_r * np.cos(g_a), g_r * np.sin(g_a))

    # Current velocities, within the robot's limits
    vel = (np.random.uniform(-cfg.DWA_MAX_VX, cfg.DWA_MAX_VX),
           np.random.uniform(-cfg.DWA_MAX_VY, cfg.DWA_MAX_VY),
           np.random.uniform(-cfg.DWA_MAX_WZ, cfg.DWA_MAX_WZ))

    return {"obs": obs, "goal": goal, "vel": vel}


def make_dwa_node():
    return Automatic.__new__(Automatic)


def run_dwa(node, scene):
    # Set the current velocity and the goal
    node.vx_cur, node.vy_cur, node.wz_cur = scene["vel"]
    gx, gy = scene["goal"]

    # Run the DWA and return the result
    return node._dwa(gx, gy, scene["obs"])



def bench(node, Ms=(100, 200, 400, 800, 1600, 3200), n_scenes=30, n_warmup=3):
    print(f"{'target M':>8} {'actual M':>9} {'median':>9} {'p95':>9} {'max':>9} {'ms/100pts':>10}")

    for M in Ms:
        # Generate all scenes first, so that only the DWA itself is timed
        scenes = [make_scene(M) for _ in range(n_scenes)]
        mean_M = np.mean([len(s["obs"]) for s in scenes])

        # Warm-up calls because the first calls are slower (memory allocation, caches)
        for scene in scenes[:n_warmup]:
            run_dwa(node, scene)

        times = []
        for scene in scenes:
            t0 = time.perf_counter()
            run_dwa(node, scene)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)   # milliseconds

        median = np.median(times)
        p95 = np.percentile(times, 95)
        per_100 = median / mean_M * 100

        print(f"{M:>8} {mean_M:>9.0f} {median:>7.1f}ms {p95:>7.1f}ms {max(times):>7.1f}ms {per_100:>10.2f}")




def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=Path(__file__).parent, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def save_reference(node, path, seed, Ms=(100, 200, 400, 800), n_per_M=15):
    # Reseed: the reference cases must not depend on what ran before (e.g. the timing)
    np.random.seed(seed)

    scenes = [make_scene(M) for M in Ms for _ in range(n_per_M)]

    obs_list, goals, vels, cmds, scores = [], [], [], [], []
    for scene in scenes:
        cmd, score = run_dwa(node, scene)
        obs_list.append(scene["obs"])
        goals.append(scene["goal"])
        vels.append(scene["vel"])
        cmds.append(cmd if cmd is not None else (np.nan, np.nan, np.nan))
        scores.append(score)   # -inf when cmd is None

    obs_len = np.array([len(o) for o in obs_list])
    obs_start = np.concatenate([[0], np.cumsum(obs_len)[:-1]])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs=np.vstack(obs_list),
        obs_start=obs_start,
        obs_len=obs_len,
        goal=np.array(goals),
        vel=np.array(vels),
        cmd=np.array(cmds, dtype=float),
        score=np.array(scores, dtype=float),
        config_keys=np.array(CONFIG_KEYS),
        config_values=np.array([float(getattr(cfg, k)) for k in CONFIG_KEYS]),
        seed=seed,
        commit=git_commit(),
    )

    n_none = sum(1 for c in cmds if np.isnan(c[0]))
    print(f"Saved {len(scenes)} cases ({n_none} with no valid command) to {path}")



def main():
    parser = argparse.ArgumentParser(description="Benchmark the DWA planner on synthetic scenes.")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed (same seed = same scenes)")
    parser.add_argument("--n-scenes", type=int, default=30,
                        help="scenes timed per value of M")
    parser.add_argument("--save-reference", nargs="?", const=str(DEFAULT_REFERENCE_PATH),
                        default=None, metavar="PATH",
                        help=f"also save the golden file (default path: {DEFAULT_REFERENCE_PATH})")
    args = parser.parse_args()

    node = make_dwa_node()

    # Set up the random seed (so we can test the same scenarios each time if we want to)
    np.random.seed(args.seed)
    bench(node, n_scenes=args.n_scenes)

    if args.save_reference:
        save_reference(node, args.save_reference, args.seed)


if __name__ == "__main__":
    main()