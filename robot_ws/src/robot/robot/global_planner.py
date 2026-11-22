"The A* global planner for our robot"
from robot.steady_node import SteadyNode
import rclpy
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import Pose2D, Point
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
import numpy as np
from robot.robot_config import DEFAULT_GOAL_X, DEFAULT_GOAL_Y

map_qos = QoSProfile(depth=1,
                     reliability=QoSReliabilityPolicy.RELIABLE,
                     durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


class Global_Planner(SteadyNode):

    def __init__(self):
        super().__init__("global_planner")

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0

        self.goal_x = DEFAULT_GOAL_X
        self.goal_y = DEFAULT_GOAL_Y

        self.grid = None
        self.map_res = 0.0
        self.map_origin_x, self.map_origin_y = 0.0, 0.0

        self.create_subscription (Pose2D, '/robot/pos', self.pos_callback, 10)
        self.create_subscription (Point, '/robot/automatic_goal', self.goal_callback, 10)
        self.create_subscription (OccupancyGrid, '/robot/map', self.map_callback, map_qos)
        self.pub_path = self.create_publisher (Path, '/robot/nav_path', 10)


    def pos_callback(self, msg: Pose2D):
        self.robot_x, self.robot_y, self.robot_theta = msg.x, msg.y, msg.theta

    def goal_callback(self, msg: Point):
        self.goal_x, self.goal_y = msg.x, msg.y

    def map_callback(self, msg: OccupancyGrid):
        self.grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self.map_res = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y
    
def main():
    rclpy.init()
    node = Global_Planner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()