import math
import time
import numpy as np
import rclpy
from robot.steady_node import SteadyNode
from robot.dwa import dwa_plan
from msgs.msg import Lidar, RPMs
from geometry_msgs.msg import Pose2D, Point
from std_msgs.msg import Bool
from nav_msgs.msg import Path
from robot.planner_utils import short_targ

from robot.robot_config import (
    MAX_RPM, MIN_RPM, LIDAR_MIN_DIST, LIDAR_MAX_RANGE_M,
    DEFAULT_GOAL_X, DEFAULT_GOAL_Y, AUTO_LOOP_HZ,
    DWA_MAX_VX, DWA_MAX_VY, DWA_MAX_WZ, DWA_MAX_ACC_V, DWA_MAX_ACC_W,
    DWA_V_SAMPLES, DWA_W_SAMPLES,
    DWA_SIM_TIME, DWA_SIM_STEPS,
    DWA_ROBOT_RADIUS, DWA_OBSTACLE_MARGIN, DWA_LETHAL_DIST,
    DWA_W_HEADING, DWA_W_CLEARANCE, DWA_W_VELOCITY, DWA_W_GOAL_DIST,
    NAV_GOAL_DIST_TOL, NAV_GOAL_ANGLE_TOL,
    OBSTACLE_MEMORY_SEC, LIDAR_STALE_SEC, STUCK_COUNT_THRESH,
    RECOVERY_BACKUP_S, RECOVERY_ROTATE_S, RECOVERY_VX, RECOVERY_WZ
)
LOOKAHEAD = 0.6
TIME_LIMIT = 3.0

# Fonctions utilitaires
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class Automatic(SteadyNode):

    def __init__(self):
        super().__init__('automatic')

        # -- Abonnements --
        self.create_subscription(Lidar,  '/robot/lidar_obstacles',  self.lidar_callback,  1)
        self.create_subscription(Pose2D, '/robot/pos',               self.pos_callback,    10)
        self.create_subscription(Point,  '/robot/automatic_goal',    self.goal_callback,   10)
        self.create_subscription(Bool,   '/robot/enable_auto',       self.enable_auto_callback, 10)
        self.create_subscription (Path, '/robot/nav_path', self.path_callback, 10)

        # -- Publication --
        self.pub_cmd = self.create_publisher(RPMs, '/robot/command', 10)

        # Etat robot
        self.is_auto      = False
        self.goal_reached = False

        # Position estimée (Odométrie + Lidar)
        self.robot_x     : float = 0.0
        self.robot_y     : float = 0.0
        self.robot_theta : float = 0.0

        # Consignes de vitesse actuelles
        self.vx_cur = 0.0
        self.vy_cur = 0.0
        self.wz_cur = 0.0

        self.goal_x : float = DEFAULT_GOAL_X
        self.goal_y : float = DEFAULT_GOAL_Y

        # Mémoire obstacles (repère MONDE + moment)
        # Stocker en repère monde est important : si on stockait en repère
        # robot, les vieux points deviendraient faux dès que le robot tourne,
        # faisant croire à DWA qu'il est cerné d'obstacles fantômes
        self._obs_memory       : list[tuple[float, float, float]] = []
        self._lidar_last_stamp : float = 0.0

        # Machine à états de récupération
        # Etats : "navigate" | "backup" | "rotate"
        self._state       = "navigate"
        self._stuck_count = 0
        self._phase_start : float = 0.0
        # Le signe de rotation alterne à chaque récupération pour éviter
        # de toujours tourner du même côté.
        self._recovery_wz_sign : float = 1.0

        self.path = None
        self.path_stamp = 0.0

        # Timer de la boucle d'asservissement
        self.create_timer(1.0 / AUTO_LOOP_HZ, self.control_loop)
        self.get_logger().info(
            f"Noeud Automatic (DWA) lancé — cible : ({self.goal_x:.2f}, {self.goal_y:.2f})"
        )

    # -- Callbacks --

    # Gestion du passage manuel/auto
    def enable_auto_callback(self, msg: Bool):
        if self.is_auto == msg.data:
            return
        if self.is_auto:
            self._stop()
            self.is_auto = False
            self._state  = "navigate"
            self._stuck_count = 0
            self.get_logger().info("Mode MANUEL activé.")
        else:
            self.is_auto      = True
            self.goal_reached = False
            self.get_logger().info("Mode AUTOMATIQUE activé.")

    def goal_callback(self, msg: Point):
        self.set_goal(msg.x, msg.y)

    # Transformation des points Lidar du repère local vers le repère monde
    def lidar_callback(self, msg: Lidar):
        now = time.monotonic()
        self._lidar_last_stamp = now

        rx, ry, rt   = self.robot_x, self.robot_y, self.robot_theta
        cos_t, sin_t = math.cos(rt), math.sin(rt)

        # Matrice de rotation 2D + translation
        new_pts = [
            (rx + d * (math.cos(a) * cos_t - math.sin(a) * sin_t),
             ry + d * (math.cos(a) * sin_t + math.sin(a) * cos_t),
             now)
            for a, d in zip(msg.angles, msg.distances)
            if math.isfinite(d) and d > LIDAR_MIN_DIST
        ]

        # Rafraichissement de la memoire tampon
        cutoff = now - OBSTACLE_MEMORY_SEC
        self._obs_memory = [p for p in self._obs_memory if p[2] > cutoff]
        self._obs_memory.extend(new_pts)

    def pos_callback(self, msg: Pose2D):
        self.robot_x     = msg.x
        self.robot_y     = msg.y
        self.robot_theta = msg.theta

    def path_callback(self, msg: Path):
        if len(msg.poses) == 0:
            return
        
        self.path = np.array([(p.pose.position.x, p.pose.position.y) for p in msg.poses])
        self.path_stamp = time.monotonic()

    # -- Boucle principale --

    def control_loop(self):
        if not self.is_auto:
            return
        if self.goal_reached:
            self._stop()
            return

        # Récupération : recul
        if self._state == "backup":
            if time.monotonic() - self._phase_start < RECOVERY_BACKUP_S:
                self._publish_command(RECOVERY_VX, 0.0, 0.0)
                return
            # Recul terminé → on passe à la rotation
            self._state       = "rotate"
            self._phase_start = time.monotonic()

        # Récupération : rotation
        if self._state == "rotate":
            if time.monotonic() - self._phase_start < RECOVERY_ROTATE_S:
                self._publish_command(0.0, 0.0, self._recovery_wz_sign * RECOVERY_WZ)
                return
            # Rotation terminée -> retour à la navigation
            self._state       = "navigate"
            self._stuck_count = 0
            self.get_logger().info("Récupération terminée — reprise de la navigation.")

        # Avertissement lidar périmé
        lidar_age = time.monotonic() - self._lidar_last_stamp
        if self._lidar_last_stamp > 0 and lidar_age > LIDAR_STALE_SEC:
            self.get_logger().warn(f"Lidar périmé ({lidar_age:.2f}s)")

        tx, ty = self.goal_x, self.goal_y
        if (self.path is not None and
            time.monotonic() - self.path_stamp < TIME_LIMIT and
            math.hypot(self.path[-1, 0] - self.goal_x, self.path[-1, 1] - self.goal_y) < 0.2):
            tx, ty = short_targ(self.path, self.robot_x, self.robot_y, LOOKAHEAD)
            

        # Cible atteinte ?
        dist_to_goal = math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)
        if dist_to_goal < NAV_GOAL_DIST_TOL:
            self.get_logger().info("Cible atteinte !")
            self.goal_reached = True
            self._stop()
            return

        

        # But en repère robot
        dx, dy       = tx - self.robot_x, ty - self.robot_y
        cos_t, sin_t = math.cos(self.robot_theta), math.sin(self.robot_theta)
        goal_lx      =  dx * cos_t + dy * sin_t
        goal_ly      = -dx * sin_t + dy * cos_t

        # Récupération des obstacles filtrés
        obs = self._get_obstacle_array()

        # Vérification d'impact immédiat ( sécurité avant DWA)
        if obs.shape[0] > 0:
            if float(np.hypot(obs[:, 0], obs[:, 1]).min()) < DWA_ROBOT_RADIUS:
                self.get_logger().warn("Obstacle immédiat — récupération d'urgence")
                self._start_recovery()
                return

        # Calcul de la commande optimale via DWA
        best_cmd, _ = dwa_plan((self.vx_cur, self.vy_cur, self.wz_cur), (goal_lx, goal_ly), obs)

        # Update de la vitesse courante pour la prochaine itération (fenêtre glissante)
        if best_cmd is not None:
            self.vx_cur, self.vy_cur, self.wz_cur = best_cmd
        else:
            self.vx_cur, self.vy_cur, self.wz_cur = 0.0, 0.0, 0.0

        if best_cmd is None:
            self._stuck_count += 1
            self.get_logger().warn(
                f"DWA : aucune trajectoire ({self._stuck_count}/{STUCK_COUNT_THRESH})"
            )
            self._stop()
            if self._stuck_count >= STUCK_COUNT_THRESH:
                self._start_recovery()
        else:
            self._stuck_count = 0
            self._publish_command(*best_cmd)

    # Alterne le sens de rotation pour maximiser les chances de dégagement
    def _start_recovery(self):
        self._recovery_wz_sign *= -1.0
        self._state       = "backup"
        self._phase_start = time.monotonic()
        self._stuck_count = 0
        self.get_logger().warn(
            f"RECOVERY : recul {RECOVERY_BACKUP_S:.1f}s puis rotation "
            f"{'droite' if self._recovery_wz_sign > 0 else 'gauche'} {RECOVERY_ROTATE_S:.1f}s"
        )

    # Reprojette les points mémorisés en local pour l'algorithme DWA
    def _get_obstacle_array(self) -> np.ndarray:
        if not self._obs_memory:
            return np.empty((0, 2), dtype=np.float64)

        rx, ry, rt   = self.robot_x, self.robot_y, self.robot_theta
        cos_t, sin_t = math.cos(rt), math.sin(rt)

        pts = np.array(
            [( (wx - rx) * cos_t + (wy - ry) * sin_t,
              -(wx - rx) * sin_t + (wy - ry) * cos_t)
             for wx, wy, _ in self._obs_memory],
            dtype=np.float64
        )
        
        # Echantillonnage adaptatif ; on garde tous les points proches, on réduit au loin
        dists      = np.hypot(pts[:, 0], pts[:, 1])
        close_mask = dists < (DWA_ROBOT_RADIUS + DWA_OBSTACLE_MARGIN) * 2.0
        close_pts  = pts[close_mask]
        far_pts    = pts[~close_mask][::2]

        return np.vstack([close_pts, far_pts]) if (close_pts.shape[0] + far_pts.shape[0]) > 0 else np.empty((0, 2), dtype=np.float64)


    #  Publication

    # Calcul du mixage moteur pour châssis Mecanum (Cinématique inverse)
    def _publish_command(self, vx: float, vy: float, w: float):
        vx_n = _clamp(vx / DWA_MAX_VX, -1.0, 1.0)
        vy_n = _clamp(vy / DWA_MAX_VY, -1.0, 1.0)
        w_n  = _clamp(w  / DWA_MAX_WZ, -1.0, 1.0)

        # Equations de distribution pour 4 roues Mecanum
        fl = vx_n - vy_n - w_n
        fr = vx_n + vy_n + w_n
        rr = vx_n - vy_n + w_n
        rl = vx_n + vy_n - w_n

        # Normalisation pour ne pas saturer les moteurs
        m = max(abs(fl), abs(fr), abs(rr), abs(rl), 1.0)
        fl, fr, rr, rl = fl/m, fr/m, rr/m, rl/m

        # Mapping vitesse linéaire -> RPM avec zone morte
        def to_rpm(n):
            if abs(n) < 0.02: return 0.0
            return math.copysign(MIN_RPM + (MAX_RPM - MIN_RPM) * abs(n), n)

        cmd = RPMs()
        cmd.front_left_rpm  = float(to_rpm(fl))
        cmd.front_right_rpm = float(to_rpm(fr))
        cmd.back_right_rpm  = float(to_rpm(rr))
        cmd.back_left_rpm   = float(to_rpm(rl))
        self.pub_cmd.publish(cmd)

    # Consigne de vitesse nulle sur tous les actionneurs
    def _stop(self):
        self.vx_cur = self.vy_cur = self.wz_cur = 0.0
        cmd = RPMs()
        cmd.front_left_rpm = cmd.front_right_rpm = \
        cmd.back_right_rpm = cmd.back_left_rpm   = 0.0
        self.pub_cmd.publish(cmd)

    #  Public
    def set_goal(self, x: float, y: float):
        self.path = None
        self.goal_x       = x
        self.goal_y       = y
        self.goal_reached = False
        self._state       = "navigate"
        self._stuck_count = 0
        self.get_logger().info(f"Nouvelle cible (DWA) : ({x:.2f}, {y:.2f})")


# -- Point d'entrée --
def main():
    rclpy.init()
    node = Automatic()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()