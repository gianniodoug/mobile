"""
Banc de mesure de la caractéristique statique P_i : PWM -> RPM des 4 roues.
Remplace arduino/src/wheelBenchmark.cpp : les encodeurs sont maintenant lus par la Raspberry Pi.

Pour chaque palier de PWM (de pwm_start à pwm_end par pas de pwm_step) :
    1. envoie la même PWM aux 4 roues via /robot/raw_command (comme la commande setpwm) ;
    2. attend settle_time secondes que les roues se stabilisent ;
    3. moyenne /robot/encoders pendant measure_time secondes.
À la fin : arrête les roues, écrit un CSV et affiche l'ajustement des polynômes (wheel_curve_fit.py).

ATTENTION : le robot doit être SUR CALES (roues en l'air), la rampe dure plusieurs minutes.
Les nœuds control, serial et encoders doivent tourner.

Usage :
    ros2 run robot wheel_benchmark
    ros2 run robot wheel_benchmark --ros-args -p direction:=backward
    ros2 run robot wheel_benchmark --ros-args -p pwm_step:=5 -p output:=/home/pi/mesure.csv
"""
import csv
import os
import time
from datetime import datetime

import numpy as np
import rclpy
from msgs.msg import RPMs, WheelSpeeds
from robot.steady_node import SteadyNode
from robot.wheel_curve_fit import WHEELS, fit_all_wheels, format_report


class WheelBenchmark(SteadyNode):

    def __init__(self):
        super().__init__("wheel_benchmark")

        self.declare_parameter("pwm_start", 40)
        self.declare_parameter("pwm_end", 254)
        self.declare_parameter("pwm_step", 2)
        self.declare_parameter("settle_time", 0.5)
        self.declare_parameter("measure_time", 1.0)
        self.declare_parameter("direction", "forward")
        self.declare_parameter("output", "")

        def param(name):
            return self.get_parameter(name).value

        self.direction = param("direction")
        if self.direction not in ("forward", "backward"):
            self.get_logger().error("direction must be forward or backward")
            raise SystemExit(1)
        self.sign = 1 if self.direction == "forward" else -1
        self.pwms = list(range(int(param("pwm_start")), int(param("pwm_end")) + 1, int(param("pwm_step"))))
        self.settle_time = float(param("settle_time"))
        self.measure_time = float(param("measure_time"))
        self.output = param("output") or os.path.expanduser(
            f"~/wheel_benchmark_{self.direction}_{datetime.now():%Y%m%d_%H%M%S}.csv")

        self.pub_raw_cmd = self.create_publisher(WheelSpeeds, "/robot/raw_command", 10)
        self.create_subscription(RPMs, "/robot/encoders", self.encoders_callback, 10)

        self.index = 0
        self.phase = "settle"
        self.phase_start = time.time()
        self.samples: list[list[float]] = []
        self.rows: list[list[float]] = []
        self.done = False

        duration = len(self.pwms) * (self.settle_time + self.measure_time)
        self.get_logger().info(
            f"Benchmark {self.direction} : {len(self.pwms)} paliers de PWM {self.pwms[0]} à {self.pwms[-1]}, "
            f"durée ~{duration / 60:.1f} min. Le robot doit être sur cales !"
        )
        self.send_pwm(self.sign * self.pwms[0])
        self.create_timer(0.02, self.loop)

    def send_pwm(self, pwm: int):
        msg = WheelSpeeds()
        msg.front_left_wheel_speed = pwm
        msg.front_right_wheel_speed = pwm
        msg.back_right_wheel_speed = pwm
        msg.back_left_wheel_speed = pwm
        self.pub_raw_cmd.publish(msg)

    def encoders_callback(self, msg: RPMs):
        if self.phase == "measure":
            self.samples.append([msg.front_left_rpm, msg.front_right_rpm, msg.back_right_rpm, msg.back_left_rpm])

    def loop(self):
        if self.done:
            return
        now = time.time()
        elapsed = now - self.phase_start

        if self.phase == "settle" and elapsed >= self.settle_time:
            self.phase = "measure"
            self.phase_start = now
            self.samples = []

        elif self.phase == "measure" and elapsed >= self.measure_time:
            pwm = self.sign * self.pwms[self.index]
            if not self.samples:
                self.get_logger().error("Aucune donnée sur /robot/encoders : le noeud encoders tourne-t-il ?")
                self.finish()
                return
            mean = np.mean(self.samples, axis=0)
            self.rows.append([pwm, *mean])
            self.get_logger().info(
                f"[{self.index + 1}/{len(self.pwms)}] PWM {pwm:4d} -> RPM "
                + "  ".join(f"{w}={v:6.1f}" for w, v in zip(WHEELS, mean))
            )

            self.index += 1
            if self.index >= len(self.pwms):
                self.finish()
                return
            self.send_pwm(self.sign * self.pwms[self.index])
            self.phase = "settle"
            self.phase_start = now

    def finish(self):
        self.send_pwm(0)
        self.done = True
        if not self.rows:
            return

        with open(self.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pwm", *WHEELS])
            writer.writerows(self.rows)
        self.get_logger().info(f"Mesures enregistrées dans {self.output}")

        rows = np.array(self.rows)
        try:
            report = format_report(fit_all_wheels(rows[:, 0], rows[:, 1:]), self.direction)
        except ValueError as e:
            self.get_logger().error(f"Ajustement impossible : {e}")
            return
        with open(os.path.splitext(self.output)[0] + "_fit.txt", "w") as f:
            f.write(report + "\n")
        self.get_logger().info("\n" + report)


def main():
    rclpy.init()
    node = WheelBenchmark()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("Interrompu : arrêt des roues")
    finally:
        node.send_pwm(0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
