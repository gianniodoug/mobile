"""
Retour d'état modal avec action intégrale pour les 4 roues.

Toutes les notations suivent Documentation/asservissement_retour_etat.pdf :
    n   : vitesses mesurées des roues (RPM)          ordre FL, FR, BR, BL
    n_c : consignes des roues (RPM)
    z   = T n, z_c = T n_c : vitesses modales         ordre x, y, theta, t
    q   : intégrale de l'erreur modale (= position, RPM.s)
    rho = -K1 z - K2 q + M z_c : commande modale     (M = I + K1)
    r   = T^T rho : commande des roues (RPM)
    u   = sign(r) P^-1(|r|) : PWM envoyée à l'arduino

Ce module ne dépend pas de ROS pour pouvoir être testé hors du robot.
"""
import math
from typing import Callable, Sequence

import numpy as np

MODES = ("x", "y", "theta", "t")
MODE_INDEX = {name: i for i, name in enumerate(MODES)}
X, Y, THETA, TWIST = range(4)

# Matrice de passage roues -> modes (orthogonale : T^T T = I)
#                    FL  FR  BR  BL
T = 0.5 * np.array([[ 1,  1,  1,  1],   # x     (avance)
                    [-1,  1, -1,  1],   # y     (latéral)
                    [-1,  1,  1, -1],   # theta (rotation)
                    [ 1,  1, -1, -1]])  # t     (torsion)


def angle_wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def gains_from_poles(omega0: float, xi: float, tau: float) -> tuple[float, float]:
    """Placement de pôles d'un mode : chi(s) = s^2 + 2 xi omega0 s + omega0^2 (pdf, section « Placement de pôles »)."""
    k1 = 2.0 * xi * omega0 * tau - 1.0
    k2 = omega0 ** 2 * tau
    return k1, k2


class StateFeedbackController:

    def __init__(self,
                 k1: Sequence[float],
                 k2: Sequence[float],
                 rpm_to_pwm: Sequence[Callable[[float], float]],
                 wheel_radius: float,
                 half_track: float,
                 use_pose: bool = True,
                 max_integral_correction: float = 40.0,
                 rpm_deadband: float = 0.5) -> None:
        """
        k1, k2       : gains par mode (ordre x, y, theta, t)
        rpm_to_pwm   : pour chaque roue, P_i^-1 : RPM >= 0 -> PWM >= 0 (inverse du polynôme)
        wheel_radius : R (m)
        half_track   : L = LX + LY (m)
        """
        self.k1 = np.array(k1, dtype=float)
        self.k2 = np.array(k2, dtype=float)
        self.rpm_to_pwm = list(rpm_to_pwm)
        self.use_pose = use_pose
        self.max_integral_correction = max_integral_correction
        self.rpm_deadband = rpm_deadband

        # Conversions vitesses physiques -> vitesses modales (pdf, section « La matrice T »)
        self.c_v = 2.0 / wheel_radius * 30.0 / math.pi                    # RPM.s/m
        self.c_theta = 2.0 * half_track / wheel_radius * 30.0 / math.pi   # RPM.s/rad

        self.q = np.zeros(4)
        # Référence de la ligne droite (x0, y0, theta0, q_y0) quand q_theta et q_y viennent de la pose
        self.pose_ref: tuple[float, float, float, float] | None = None

        # Dernières valeurs calculées, pour le debug
        self.z = np.zeros(4)
        self.z_c = np.zeros(4)
        self.rho = np.zeros(4)
        self.r = np.zeros(4)
        self.saturated = False
        self.using_pose = False

    def reset(self) -> None:
        self.q[:] = 0.0
        self.pose_ref = None

    def set_gain(self, mode: str, k1: float | None = None, k2: float | None = None) -> None:
        i = MODE_INDEX[mode]
        if k1 is not None:
            self.k1[i] = k1
        if k2 is not None:
            self.k2[i] = k2

    def compute(self,
                n_c: Sequence[float],
                n: Sequence[float],
                dt: float,
                pose: tuple[float, float, float] | None = None) -> list[int]:
        """
        Calcule les 4 PWM (FL, FR, BR, BL) à envoyer.
        pose : (x, y, theta) récente de /robot/pos, ou None si indisponible.
        """
        n_c = np.asarray(n_c, dtype=float)
        n = np.asarray(n, dtype=float)

        # Consigne nulle : on arrête les moteurs et on oublie l'erreur accumulée
        if not np.any(n_c):
            self.reset()
            self.z = T @ n
            self.z_c = np.zeros(4)
            self.rho = np.zeros(4)
            self.r = np.zeros(4)
            self.saturated = False
            self.using_pose = False
            return [0, 0, 0, 0]

        z_c = T @ n_c
        z = T @ n

        # Action intégrale : q_{k+1} = q_k + dt (z_k - z_c)
        q_new = self.q + dt * (z - z_c)

        # En ligne droite (z_c,y = z_c,theta = 0), q_theta et q_y sont lus sur la pose (pdf, section « Mesurer q_theta et q_y avec la pose »)
        straight = abs(z_c[Y]) < 1e-6 and abs(z_c[THETA]) < 1e-6
        self.using_pose = self.use_pose and straight and pose is not None
        if self.using_pose:
            x, y, theta = pose
            if self.pose_ref is None:
                # Référence choisie pour que q reste continu au moment où l'on bascule sur la pose
                self.pose_ref = (x, y, theta - self.q[THETA] / self.c_theta, self.q[Y])
            x0, y0, theta0, q_y0 = self.pose_ref
            d = -math.sin(theta0) * (x - x0) + math.cos(theta0) * (y - y0)
            q_new[THETA] = self.c_theta * angle_wrap(theta - theta0)
            q_new[Y] = q_y0 + self.c_v * d
        else:
            self.pose_ref = None

        # Le terme intégral ne peut pas corriger de plus de max_integral_correction RPM par mode
        # (protège des sauts de pose de l'ICP et de l'emballement)
        with np.errstate(divide="ignore"):
            q_max = np.where(self.k2 > 0, self.max_integral_correction / self.k2, np.inf)
        q_new = np.clip(q_new, -q_max, q_max)

        # Retour d'état + anticipation : rho = -K1 z - K2 q + (I + K1) z_c
        rho = -self.k1 * z - self.k2 * q_new + (1.0 + self.k1) * z_c
        r = T.T @ rho

        # Linéarisation : u_i = sign(r_i) P_i^-1(|r_i|)
        pwm = []
        for i in range(4):
            if abs(r[i]) < self.rpm_deadband:
                pwm.append(0.0)
            else:
                pwm.append(math.copysign(self.rpm_to_pwm[i](abs(r[i])), r[i]))
        pwm = np.array(pwm)

        # Anti-windup : si une roue sature, on n'intègre pas les modes calculés aux encodeurs
        self.saturated = bool(np.any(np.abs(pwm) >= 255))
        if self.saturated:
            integrated = [X, TWIST] if self.using_pose else [X, Y, THETA, TWIST]
            q_new[integrated] = self.q[integrated]
        self.q = q_new

        self.z, self.z_c, self.rho, self.r = z, z_c, rho, r
        return [int(round(p)) for p in np.clip(pwm, -255, 255)]
