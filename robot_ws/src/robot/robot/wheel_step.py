"""
Identification de la constante de temps tau des roues (pdf, section « Identification de tau »).

Essai en BOUCLE OUVERTE : les PWM sont envoyées directement (comme setpwm), sans le correcteur,
pour mesurer la dynamique des moteurs seuls. Pour chaque roue, la PWM est calculée avec P_i^-1
(robot_config.WHEEL_PWM_TO_RPM) afin d'obtenir rpm_low puis rpm_high.

Séquence, répétée `repeats` fois :
    1. palier bas  : PWM = P_i^-1(rpm_low)  pendant hold_low secondes
    2. échelon     : PWM = P_i^-1(rpm_high) pendant hold_high secondes  <- réponse enregistrée
L'échelon part d'une vitesse non nulle pour rester hors de la zone morte (frottement sec au démarrage).

Pour chaque roue, on ajuste tau et un retard pur d sur la réponse mesurée, en tenant compte de la
moyenne glissante des encodeurs (fenêtre ENCODER_WINDOW). On vérifie aussi que la vitesse atteinte
vaut bien rpm_high, c'est-à-dire que les polynômes P_i sont à jour (gain statique = 1).

ATTENTION : le robot doit être SUR CALES (roues en l'air). Les nœuds control, serial et encoders doivent tourner.

Usage :
    ros2 run robot wheel_step
    ros2 run robot wheel_step --ros-args -p rpm_low:=60.0 -p rpm_high:=120.0 -p repeats:=3
"""
import csv
import math
import os
import time
from datetime import datetime

import numpy as np
import rclpy
from msgs.msg import RPMs, WheelSpeeds
from robot.robot_config import ENCODER_WINDOW, WHEEL_PWM_TO_RPM
from robot.steady_node import SteadyNode
from robot.wheel_curve_fit import WHEELS, rpm_to_pwm


def measured_step(t: np.ndarray, tau: float, delay: float, window: float) -> np.ndarray:
    """
    Réponse normalisée (0 -> 1) d'un premier ordre de constante tau, retardée de delay,
    vue à travers une moyenne glissante de largeur window.
    """
    s = np.clip(t - delay, 0.0, None)
    early = (s - tau * (1.0 - np.exp(-s / tau))) / window                                 # s < window
    late = 1.0 - (tau / window) * (np.exp(-(s - window) / tau) - np.exp(-s / tau))       # s >= window
    return np.where(s < window, early, late)


def fit_tau(t: np.ndarray, y: np.ndarray, window: float) -> tuple[float, float, float]:
    """Moindres carrés sur une grille (tau, delay). Renvoie tau, delay, RMSE (normalisée)."""
    best = (math.inf, 0.0, 0.0)
    for tau in np.arange(0.01, 0.60, 0.002):
        for delay in np.arange(0.0, 0.10, 0.005):
            err = float(np.sqrt(np.mean((measured_step(t, tau, delay, window) - y) ** 2)))
            if err < best[0]:
                best = (err, tau, delay)
    return best[1], best[2], best[0]


class WheelStep(SteadyNode):

    def __init__(self):
        super().__init__("wheel_step")

        self.declare_parameter("rpm_low", 60.0)
        self.declare_parameter("rpm_high", 120.0)
        self.declare_parameter("hold_low", 2.0)
        self.declare_parameter("hold_high", 2.0)
        self.declare_parameter("repeats", 3)
        self.declare_parameter("output", "")

        def param(name):
            return self.get_parameter(name).value

        self.rpm_low = float(param("rpm_low"))
        self.rpm_high = float(param("rpm_high"))
        self.hold_low = float(param("hold_low"))
        self.hold_high = float(param("hold_high"))
        self.repeats = int(param("repeats"))
        self.output = param("output") or os.path.expanduser(f"~/wheel_step_{datetime.now():%Y%m%d_%H%M%S}.csv")

        # PWM de chaque roue, calculées avec P_i^-1 (ordre FL, FR, BR, BL)
        self.pwm_low = [round(rpm_to_pwm(WHEEL_PWM_TO_RPM[w], self.rpm_low)) for w in WHEELS]
        self.pwm_high = [round(rpm_to_pwm(WHEEL_PWM_TO_RPM[w], self.rpm_high)) for w in WHEELS]

        self.pub_raw_cmd = self.create_publisher(WheelSpeeds, "/robot/raw_command", 10)
        self.create_subscription(RPMs, "/robot/encoders", self.encoders_callback, 10)

        self.repeat = 0
        self.phase = "low"
        self.phase_start = time.time()
        self.samples: list[list[float]] = []    # [repeat, phase, t depuis l'échelon, FL, FR, BR, BL]
        self.done = False

        self.get_logger().info(
            f"Echelon {self.rpm_low:.0f} -> {self.rpm_high:.0f} RPM, {self.repeats} fois. "
            f"PWM basses {self.pwm_low}, PWM hautes {self.pwm_high} (FL, FR, BR, BL). Le robot doit être sur cales !"
        )
        self.send_pwm(self.pwm_low)
        self.create_timer(0.01, self.loop)

    def send_pwm(self, pwms: list[int]):
        msg = WheelSpeeds()
        msg.front_left_wheel_speed = int(pwms[0])
        msg.front_right_wheel_speed = int(pwms[1])
        msg.back_right_wheel_speed = int(pwms[2])
        msg.back_left_wheel_speed = int(pwms[3])
        self.pub_raw_cmd.publish(msg)

    def encoders_callback(self, msg: RPMs):
        if self.done:
            return
        t = time.time() - self.phase_start
        if self.phase == "low":
            t -= self.hold_low    # temps négatif : avant l'échelon
        self.samples.append([self.repeat, self.phase == "high", t,
                             msg.front_left_rpm, msg.front_right_rpm, msg.back_right_rpm, msg.back_left_rpm])

    def loop(self):
        if self.done:
            return
        elapsed = time.time() - self.phase_start
        if self.phase == "low" and elapsed >= self.hold_low:
            self.send_pwm(self.pwm_high)
            self.phase = "high"
            self.phase_start = time.time()
        elif self.phase == "high" and elapsed >= self.hold_high:
            self.get_logger().info(f"Echelon {self.repeat + 1}/{self.repeats} enregistré")
            self.repeat += 1
            if self.repeat >= self.repeats:
                self.finish()
                return
            self.send_pwm(self.pwm_low)
            self.phase = "low"
            self.phase_start = time.time()

    def finish(self):
        self.send_pwm([0, 0, 0, 0])
        self.done = True

        with open(self.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["repeat", "after_step", "t", *WHEELS])
            writer.writerows(self.samples)
        self.get_logger().info(f"Mesures enregistrées dans {self.output}")

        report = analyse(np.array(self.samples), self.rpm_high, self.hold_low, self.hold_high)
        with open(os.path.splitext(self.output)[0] + "_tau.txt", "w") as f:
            f.write(report + "\n")
        self.get_logger().info("\n" + report)


def analyse(samples: np.ndarray, rpm_high: float, hold_low: float, hold_high: float,
            window: float = ENCODER_WINDOW) -> str:
    """samples : colonnes [repeat, after_step, t, FL, FR, BR, BL]."""
    if len(samples) == 0:
        return "Aucune donnée sur /robot/encoders : le noeud encoders tourne-t-il ?"

    lines = [f"Identification de tau (fenêtre encodeurs h = {window} s)", "",
             "roue         tau (s)   retard (s)   t63 - h/2 (s)   vitesse finale   écart / consigne   RMSE fit",
             ]
    taus = []
    for i, w in enumerate(WHEELS):
        t_all, y_all, finals = [], [], []
        for rep in np.unique(samples[:, 0]):
            s = samples[samples[:, 0] == rep]
            before = s[(s[:, 1] == 0) & (s[:, 2] > -0.5)][:, 3 + i]          # dernière 0,5 s avant l'échelon
            after = s[s[:, 1] == 1]
            final = after[after[:, 2] > hold_high - 0.5][:, 3 + i]          # dernière 0,5 s après l'échelon
            if len(before) == 0 or len(final) == 0:
                continue
            n0, n_inf = np.mean(before), np.mean(final)
            t_all.append(after[:, 2])
            y_all.append((after[:, 3 + i] - n0) / (n_inf - n0))
            finals.append(n_inf)
        if not t_all:
            lines.append(f"{w:11s}  pas assez de données")
            continue
        t, y = np.concatenate(t_all), np.concatenate(y_all)
        tau, delay, err = fit_tau(t, y, window)
        order = np.argsort(t)
        crossed = np.nonzero(y[order] >= 0.632)[0]
        t63 = t[order][crossed[0]] - window / 2 if len(crossed) else float("nan")
        final = float(np.mean(finals))
        taus.append(tau)
        lines.append(f"{w:11s}  {tau:7.3f}   {delay:10.3f}   {t63:13.3f}   {final:11.1f} RPM   "
                     f"{100 * (final - rpm_high) / rpm_high:+14.1f} %   {err:8.3f}")

    if taus:
        lines += ["",
                  f"Écart entre roues : tau de {min(taus):.3f} à {max(taus):.3f} s",
                  "Si une vitesse finale s'écarte de plus de ~5 % de la consigne, les polynômes ne sont plus",
                  "à jour : relancer d'abord `ros2 run robot wheel_benchmark`.",
                  "",
                  "# À copier dans robot_config.py :",
                  f"WHEEL_TAU = {np.mean(taus):.3f}"]
    return "\n".join(lines)


def main():
    rclpy.init()
    node = WheelStep()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node.get_logger().info("Interrompu : arrêt des roues")
    finally:
        node.send_pwm([0, 0, 0, 0])
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
