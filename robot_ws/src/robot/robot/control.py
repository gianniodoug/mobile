import rclpy
import numpy as np
from math import copysign
from robot.steady_node import SteadyNode
from msgs.msg import WheelSpeeds, RPMs, AsservParamChange
from std_msgs.msg import Empty, String
from geometry_msgs.msg import Pose2D
from time import time
from typing import Callable
from robot.robot_config import FL_KP, FL_KI, FL_KD, FR_KP, FR_KI, FR_KD, BR_KP, BR_KI, BR_KD, BL_KP, BL_KI, BL_KD, MAX_ACCUMULATED_ERROR, WHEEL_UPDATING_PERIOD
from robot.robot_config import WHEEL_PWM_TO_RPM
from robot.robot_config import (CONTROL_MODE, WHEEL_TAU, SF_XI, SF_OMEGA0, SF_USE_POSE, SF_POSE_TIMEOUT,
                                SF_MAX_INTEGRAL_CORRECTION, SF_RPM_DEADBAND, WHEEL_RADIUS, LX, LY)
from robot.state_feedback import StateFeedbackController, MODES, gains_from_poles
from textwrap import dedent

class Control(SteadyNode):

    class WheelControl:
        # PID pour une roue
        # Ce PID n'est là que pour compenser des petites erreurs, la formule finale pour calc la PMW est
        # PWM = int(basePWM + WheelControl.calcPWM())
        def __init__(self, basePWMFunction: Callable[[float], float], kp: float, ki: float, kd: float, logger) -> None:
            self.kp = kp
            self.ki = ki
            self.kd = kd
            self.accumulated_error = 0.0
            self.last_error = 0.0
            self.max_accumulated_error = MAX_ACCUMULATED_ERROR
            self.min_accumulated_error = -MAX_ACCUMULATED_ERROR
            self.last_time = time()
            self.basePWMFunction = basePWMFunction
            self.target_rpm = 0.
            self.current_rpm = 0.
            self.logger = logger
            self.overriden_pwm: None | int = None  # Bypass asserv and just send a pwm


        def calcPWM(self) -> int:
            t = time()
            deltaTime = t - self.last_time
            self.last_time = t

            if self.overriden_pwm != None:
                return self.overriden_pwm


            error = self.target_rpm - self.current_rpm

            self.accumulated_error += error * deltaTime
            if self.accumulated_error > self.max_accumulated_error:
                self.accumulated_error = self.max_accumulated_error
            elif self.accumulated_error < self.min_accumulated_error:
                self.accumulated_error = self.min_accumulated_error

            derivative = 0
            if deltaTime != 0:
                derivative = (error - self.last_error) / deltaTime

            #pwm = int(self.basePWMFunction(self.target_rpm) + self.kp * error + self.ki * self.accumulated_error + self.kd * derivative)
            pwm = int(self.kp * error + self.ki * self.accumulated_error + self.kd * derivative)

            if pwm > 255:
                pwm = 255
            elif pwm < -255:
                pwm = -255

            return pwm


    def __init__(self):
        super().__init__("control")

        self.pub_wheels = self.create_publisher(WheelSpeeds, "/robot/wheels", 10)
        self.sub_encoders = self.create_subscription(RPMs, "/robot/encoders", self.encoders_callback, 10)
        self.sub_asserv_params_change = self.create_subscription(AsservParamChange, "robot/asserv_params", self.asserv_param_change_callback, 10)

        self.create_subscription(RPMs, "/robot/command", self.cmd_callback, 10)
        self.create_subscription(Pose2D, "/robot/pos", self.pos_callback, 10)
        self.create_subscription(WheelSpeeds, "/robot/raw_command", self.raw_cmd_callback, 10)

        self.pub_pid_print = self.create_publisher(String, "/robot/pid_print_values", 10)
        self.create_subscription(Empty, "/robot/pid_print_request", self.pid_print_request, 10)

        self.t = self.create_timer(WHEEL_UPDATING_PERIOD, self.update_wheels)

        # init asservissement
        def base_pwm(rpmfunc):
            def calculate_base_pwm(rpm):
                if rpm == 0.0:
                    return 0
                return int(copysign(self.binary_search(rpmfunc, abs(rpm)), rpm))
            return calculate_base_pwm

        self.wheel_map = {
            "frontleft": 0,
            "frontright": 1,
            "backright": 2,
            "backleft": 3,
        }

        self.wheelControls = [
            self.WheelControl(base_pwm(self.front_left_RPM), FL_KP, FL_KI, FL_KD, self.get_logger()),
            self.WheelControl(base_pwm(self.front_right_RPM), FR_KP, FR_KI, FR_KD, self.get_logger()),
            self.WheelControl(base_pwm(self.back_right_RPM), BR_KP, BR_KI, BR_KD, self.get_logger()),
            self.WheelControl(base_pwm(self.back_left_RPM), BL_KP, BL_KI, BL_KD, self.get_logger()),
        ]

        # init retour d'état modal (voir robot/state_feedback.py et Documentation/asservissement_retour_etat.pdf)
        self.control_mode = CONTROL_MODE
        gains = [gains_from_poles(SF_OMEGA0[mode], SF_XI, WHEEL_TAU) for mode in MODES]
        for mode, (k1, k2) in zip(MODES, gains):
            if k1 <= -1:
                self.get_logger().warn(f"k1 <= -1 for mode {mode} : unstable, increase omega0 (tau={WHEEL_TAU})")

        def rpm_to_pwm(rpmfunc):
            # Inverse du polynôme P_i, cherché dans la plage réelle de PWM [0, 255]
            return lambda rpm: self.binary_search(rpmfunc, rpm, high=255.0)

        self.state_feedback = StateFeedbackController(
            k1=[k1 for k1, _ in gains],
            k2=[k2 for _, k2 in gains],
            rpm_to_pwm=[rpm_to_pwm(self.front_left_RPM), rpm_to_pwm(self.front_right_RPM),
                        rpm_to_pwm(self.back_right_RPM), rpm_to_pwm(self.back_left_RPM)],
            wheel_radius=WHEEL_RADIUS,
            half_track=LX + LY,
            use_pose=SF_USE_POSE,
            max_integral_correction=SF_MAX_INTEGRAL_CORRECTION,
            rpm_deadband=SF_RPM_DEADBAND,
        )
        self.pose: tuple[float, float, float] | None = None
        self.pose_time = 0.0
        self.last_update_time = time()

        self.get_logger().info("Control Node launched")

    def cmd_callback(self, cmd_msg: RPMs):
        self.get_logger().info(
            f"Set target RPMs {cmd_msg.front_left_rpm}, {cmd_msg.front_right_rpm}, {cmd_msg.back_right_rpm}, {cmd_msg.back_left_rpm}"
        )

        for i in range(4):
            self.wheelControls[i].overriden_pwm = None

        self.wheelControls[0].target_rpm = cmd_msg.front_left_rpm
        self.wheelControls[1].target_rpm = cmd_msg.front_right_rpm
        self.wheelControls[2].target_rpm = cmd_msg.back_right_rpm
        self.wheelControls[3].target_rpm = cmd_msg.back_left_rpm

    def pos_callback(self, msg: Pose2D):
        self.pose = (msg.x, msg.y, msg.theta)
        self.pose_time = time()

    def raw_cmd_callback(self, cmd_msg: WheelSpeeds):
        self.get_logger().info(
            f"Set target WheelSpeeds {cmd_msg.front_left_wheel_speed}, {cmd_msg.front_right_wheel_speed}, {cmd_msg.back_right_wheel_speed}, {cmd_msg.back_left_wheel_speed}"
        )

        self.wheelControls[0].overriden_pwm = cmd_msg.front_left_wheel_speed
        self.wheelControls[1].overriden_pwm = cmd_msg.front_right_wheel_speed
        self.wheelControls[2].overriden_pwm = cmd_msg.back_right_wheel_speed
        self.wheelControls[3].overriden_pwm = cmd_msg.back_left_wheel_speed
        self.state_feedback.reset()


    def encoders_callback(self, msg: RPMs):
        self.wheelControls[0].current_rpm = msg.front_left_rpm
        self.wheelControls[1].current_rpm = msg.front_right_rpm
        self.wheelControls[2].current_rpm = msg.back_right_rpm
        self.wheelControls[3].current_rpm = msg.back_left_rpm

    def update_wheels(self):
        t = time()
        dt = min(t - self.last_update_time, 0.1)
        self.last_update_time = t

        overriden = any(wc.overriden_pwm is not None for wc in self.wheelControls)
        if self.control_mode == "state" and not overriden:
            pose = self.pose if t - self.pose_time < SF_POSE_TIMEOUT else None
            pwms = self.state_feedback.compute(
                n_c=[wc.target_rpm for wc in self.wheelControls],
                n=[wc.current_rpm for wc in self.wheelControls],
                dt=dt,
                pose=pose,
            )
            sf = self.state_feedback
            self.get_logger().debug(
                f"z={sf.z.round(1)} z_c={sf.z_c.round(1)} q={sf.q.round(2)} rho={sf.rho.round(1)} "
                f"r={sf.r.round(1)} pwm={pwms} pose={sf.using_pose} sat={sf.saturated}"
            )
        else:
            pwms = [wc.calcPWM() for wc in self.wheelControls]

        msg = WheelSpeeds()
        msg.front_left_wheel_speed = pwms[0]
        msg.front_right_wheel_speed = pwms[1]
        msg.back_right_wheel_speed = pwms[2]
        msg.back_left_wheel_speed = pwms[3]

        self.pub_wheels.publish(msg)

    def asserv_param_change_callback(self, msg: AsservParamChange):
        # Changement de correcteur : asserv state / asserv pid
        if msg.param_id == "mode":
            self.control_mode = "state" if msg.new_value >= 0.5 else "pid"
            self.state_feedback.reset()
            for wc in self.wheelControls:
                wc.accumulated_error = 0.0
                wc.last_error = 0.0
                wc.last_time = time()
            self.get_logger().info(f"Control mode set to {self.control_mode}")
            return

        # Gains du retour d'état : wheel_id contient le mode (x, y, theta, t)
        if msg.param_id in ("k1", "k2", "omega0"):
            if msg.wheel_id not in MODES:
                self.get_logger().error(f"Unknown mode {msg.wheel_id}, use one of {MODES}")
                return
            if msg.param_id == "k1":
                self.state_feedback.set_gain(msg.wheel_id, k1=msg.new_value)
            elif msg.param_id == "k2":
                self.state_feedback.set_gain(msg.wheel_id, k2=msg.new_value)
            else:
                k1, k2 = gains_from_poles(msg.new_value, SF_XI, WHEEL_TAU)
                self.state_feedback.set_gain(msg.wheel_id, k1=k1, k2=k2)
            self.get_logger().info(
                f"Set {msg.param_id} of mode {msg.wheel_id} to {msg.new_value} "
                f"(k1={self.state_feedback.k1.round(3)}, k2={self.state_feedback.k2.round(3)})"
            )
            return

        wheel_control = self.wheelControls[self.wheel_map[msg.wheel_id]]
        if msg.param_id == "kp":
            wheel_control.kp = msg.new_value
        elif msg.param_id == "kd":
            wheel_control.kd = msg.new_value
        elif msg.param_id == "ki":
            wheel_control.ki = msg.new_value

        self.get_logger().info(f"Set {msg.param_id} of wheel {msg.wheel_id} to {msg.new_value}")

    def pid_print_request(self, request: Empty):
        msg = String()
        msg.data = dedent(f"""
            pid values :
            Front Left : kp={self.wheelControls[0].kp}, ki={self.wheelControls[0].ki}, kd={self.wheelControls[0].kd},
            Front Right : kp={self.wheelControls[1].kp}, ki={self.wheelControls[1].ki}, kd={self.wheelControls[1].kd}
            Back Right : kp={self.wheelControls[2].kp}, ki={self.wheelControls[2].ki}, kd={self.wheelControls[2].kd}
            Back Left : kp={self.wheelControls[3].kp}, ki={self.wheelControls[3].ki}, kd={self.wheelControls[3].kd}
            Control mode : {self.control_mode}
            State feedback (modes {", ".join(MODES)}) :
            k1={self.state_feedback.k1.round(3)}, k2={self.state_feedback.k2.round(3)}
        """)
        self.get_logger().info(msg.data)
        self.pub_pid_print.publish(msg)

    # Polynômes P_i : PWM -> RPM mesurés avec wheelBenchmark.cpp. Leur inverse (binary_search) sert à la
    # linéarisation du retour d'état : u_i = P_i^-1(r_i). Le reste (match_front_right...) n'est plus utilisé
    # Coefficients dans robot_config.WHEEL_PWM_TO_RPM (mesurés avec : ros2 run robot wheel_benchmark)
    def pwm_to_rpm(self, wheel: str, speed: float) -> float:
        if speed == 0:
            return 0.0
        return float(np.polyval(WHEEL_PWM_TO_RPM[wheel], speed))

    def front_left_RPM(self, speed: int):
        return self.pwm_to_rpm("frontleft", speed)

    def front_right_RPM(self, speed: int):
        return self.pwm_to_rpm("frontright", speed)

    def back_left_RPM(self, speed: int):
        return self.pwm_to_rpm("backleft", speed)

    def back_right_RPM(self, speed: int):
        return self.pwm_to_rpm("backright", speed)

    def match_front_right(self, rpm_function, speed: int):
        if (speed == 0):
            return 0
        return copysign(self.find_matching_wheel_speed(rpm_function, self.front_right_RPM(abs(speed))), speed)

    # https://github.com/Fypur/Navigation/wiki/Encoders#software-based-drive-correction-via-polynomial-regression
    def find_matching_wheel_speed(self, rpm_function, target_RPM: float):
        """
        Finds the required speed for a specific wheel (given its rpm_function) 
        to match a target_RPM.
        """
        if (target_RPM == 0):
            return 0.

        return self.binary_search(rpm_function, target_RPM)

    def binary_search(self, func, target, high=500.0):
        # Binary Search to find the required speed
        low = 0.0
        tolerance = 1e-2

        while (high - low) > tolerance:
            mid = (low + high) / 2.0
            mid_rpm = func(mid)

            if mid_rpm < target:
                low = mid
            else:
                high = mid

        return (low + high) / 2.0

def main():
    rclpy.init()
    node = Control()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
