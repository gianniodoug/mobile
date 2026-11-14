import numpy as np
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

    best_score = -np.inf
    best_cmd   = None

    for vx in vx_arr:
        for vy in vy_arr:
            for wz in wz_arr:
                # Simulation de la trajectoire résultante
                min_cl, fx, fy, ft = _simulate_trajectory(vx, vy, wz, obs)

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
def _simulate_trajectory(vx, vy, wz, obs):
    dt = DWA_SIM_TIME / DWA_SIM_STEPS
    x, y, theta = 0.0, 0.0, 0.0
    min_clearance = np.inf

    for _ in range(DWA_SIM_STEPS):
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        # Modèle cinématique d'un robot holonome
        x    += (vx * cos_t - vy * sin_t) * dt
        y    += (vx * sin_t + vy * cos_t) * dt
        theta = _angle_wrap(theta + wz * dt)

        if obs.shape[0] > 0:
            dists = np.hypot(obs[:, 0] - x, obs[:, 1] - y)
            min_d = float(dists.min())
            if min_d < min_clearance:
                min_clearance = min_d
            # Arrêt prématuré si collision fatale détectée
            if min_d < DWA_LETHAL_DIST:
                return min_d, x, y, theta

    if obs.shape[0] == 0:
        min_clearance = LIDAR_MAX_RANGE_M

    return min_clearance, x, y, theta


# Fonctions utilitaires
def _angle_wrap(angle: float) -> float:
    while angle > math.pi:  angle -= 2 * math.pi
    while angle < -math.pi: angle += 2 * math.pi
    return angle