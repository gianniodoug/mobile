import numpy as np
from scipy.spatial import cKDTree
import math
from robot.robot_config import (
    AUTO_LOOP_HZ,
    DWA_MAX_VX, DWA_MAX_VY, DWA_MAX_WZ, DWA_MAX_ACC_V, DWA_MAX_ACC_W,
    DWA_V_SAMPLES, DWA_W_SAMPLES, DWA_SIM_TIME, DWA_SIM_STEPS,
    DWA_ROBOT_RADIUS, DWA_OBSTACLE_MARGIN, DWA_LETHAL_DIST,
    DWA_W_HEADING, DWA_W_CLEARANCE, DWA_W_VELOCITY, DWA_W_GOAL_DIST,
    LIDAR_MAX_RANGE_M,
)



# Génération et évaluation de l'espace des vitesses (Fenêtre dynamique)
def dwa_plan(cur_vel, goal: tuple, obs: np.ndarray):
    vx_cur, vy_cur, wz_cur = cur_vel
    goal_lx, goal_ly = goal

    dt_ctrl = 1.0 / AUTO_LOOP_HZ
    acc_v   = DWA_MAX_ACC_V * dt_ctrl
    acc_w   = DWA_MAX_ACC_W * dt_ctrl

    # Définition de la fenêtre dynamique basée sur les capacités d'accélération
    vx_arr = np.linspace(max(-DWA_MAX_VX, vx_cur - acc_v),
                            min( DWA_MAX_VX, vx_cur + acc_v), DWA_V_SAMPLES)
    vy_arr = np.linspace(max(-DWA_MAX_VY, vy_cur - acc_v),
                            min( DWA_MAX_VY, vy_cur + acc_v), DWA_V_SAMPLES)
    wz_arr = np.linspace(max(-DWA_MAX_WZ, wz_cur - acc_w),
                            min( DWA_MAX_WZ, wz_cur + acc_w), DWA_W_SAMPLES)

    
    vx_grid, vy_grid, wz_grid = np.meshgrid(vx_arr, vy_arr, wz_arr, indexing='ij')
    
    VX = np.ravel(vx_grid)
    VY = np.ravel(vy_grid)
    WZ = np.ravel(wz_grid)

    X, Y, TH = _simulate_trajectory(VX, VY, WZ)

    MIN_CL = _min_clearance(X, Y, obs)

    best_score = -np.inf
    best_cmd   = None


    for i in range(len(VX)):
        vx, vy, wz = VX[i], VY[i], WZ[i]

        # Final pose of candidate i
        fx = X[i, -1]
        fy = Y[i, -1]
        ft = TH[i]

        # Smallest distance to an obstacle along trajectory i
        min_cl = MIN_CL[i]

        # Élimination des trajectoires collisionnelles
        if min_cl < DWA_ROBOT_RADIUS + DWA_OBSTACLE_MARGIN:
            continue

        # Calcul des composantes de la fonction de coût (normalisées entre 0 et 1)
        heading_score   = (1.0 + math.cos(
            _angle_wrap(math.atan2(goal_ly, goal_lx) - ft))) / 2.0
        clearance_score = min(min_cl, LIDAR_MAX_RANGE_M) / LIDAR_MAX_RANGE_M
        velocity_score  = math.hypot(vx, vy) / math.hypot(DWA_MAX_VX, DWA_MAX_VY)
        final_dist      = math.hypot(goal_lx - fx, goal_ly - fy)
        max_possible    = math.hypot(goal_lx, goal_ly) + DWA_MAX_VX * DWA_SIM_TIME
        goal_dist_score = 1.0 - min(final_dist / max(max_possible, 1e-6), 1.0)

        # Somme pondérée
        score = (DWA_W_HEADING   * heading_score
               + DWA_W_CLEARANCE * clearance_score
               + DWA_W_VELOCITY  * velocity_score
               + DWA_W_GOAL_DIST * goal_dist_score)

        if score > best_score:
            best_score = score
            best_cmd   = (vx, vy, wz)

    return best_cmd, best_score



# Intégration d'Euler pour prédire la position future
def _simulate_trajectory(VX, VY, WZ):
    n = len(VX)
    dt = DWA_SIM_TIME / DWA_SIM_STEPS
    x, y, thita = np.zeros(n), np.zeros(n), np.zeros(n) 

    X = np.zeros((n, DWA_SIM_STEPS))
    Y = np.zeros((n, DWA_SIM_STEPS))

    for k in range(DWA_SIM_STEPS):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        # Modèle cinématique d'un robot holonome
        x   += (VX * cos_t - VY * sin_t) * dt
        y   += (VX * sin_t + VY * cos_t) * dt
        thita = thita + WZ * dt

        X[:,k] = x
        Y[:,k] = y

    return X, Y, thita

        

# Fonctions utilitaires
def _angle_wrap(angle: float) -> float:
    while angle > math.pi:  angle -= 2 * math.pi
    while angle < -math.pi: angle += 2 * math.pi
    return angle

# Mininum distance from object for every path
def _min_clearance(X, Y, obs):
    s = X.shape[0]

    # Set obstacles at max radius in case we have nothing close to us
    if obs.shape[0] == 0:
        return np.full(n, LIDAR_MAX_RANGE_M)

    tree = cKDTree(obs)

    x_rav = X.ravel()
    y_rav = Y.ravel()
    positions = np.column_stack((x_rav, y_rav))

    dists, _ = tree.query(positions)

    dists = dists.reshape(s, DWA_SIM_STEPS)


    return dists.min(axis=1)