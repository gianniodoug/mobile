"The A* global planner for our robot"
from robot.steady_node import SteadyNode
import rclpy
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import Pose2D, Point, PoseStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
import numpy as np
from robot.robot_config import DEFAULT_GOAL_X, DEFAULT_GOAL_Y, DWA_ROBOT_RADIUS, DWA_OBSTACLE_MARGIN 
from robot.planner_utils import make_occ, world_to_cell, cell_to_world
from robot.astar import astar
import math

map_qos = QoSProfile(depth=1,
                     reliability=QoSReliabilityPolicy.RELIABLE,
                     durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

CONSIDERED_OCCUPIED = 65

PLANNER_EXTRA = 0.15



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

        self.create_timer(1.0, self.plan_callback)

    def pos_callback(self, msg: Pose2D):
        self.robot_x, self.robot_y, self.robot_theta = msg.x, msg.y, msg.theta

    def goal_callback(self, msg: Point):
        self.goal_x, self.goal_y = msg.x, msg.y

    def map_callback(self, msg: OccupancyGrid):
        self.grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self.map_res = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y

    def plan_callback(self):
        if self.grid is None:
            return
        r = math.ceil((DWA_ROBOT_RADIUS + DWA_OBSTACLE_MARGIN + PLANNER_EXTRA) / self.map_res)
        
        occ = make_occ(self.grid, CONSIDERED_OCCUPIED, r)
        start = world_to_cell(self.robot_x, self.robot_y, self.map_origin_x, self.map_origin_y, self.map_res, self.grid.shape)
        goal = world_to_cell(self.goal_x, self.goal_y, self.map_origin_x, self.map_origin_y, self.map_res, self.grid.shape)

        if start is None:
            self.get_logger().warn("Starting point is outside the known map")
            return
        if goal is None:
            self.get_logger().warn("Goal point is outside the known map")
            return

        # that's cause when the robot starts close to a wall
        # it sees its location as occupied due to the inflation
        # so we ensure a safe start like that
        # we only remove the inflation around the robot, the real walls stay
        iy, ix = start
        y0, y1 = max(0, iy - r), iy + r + 1
        x0, x1 = max(0, ix - r), ix + r + 1
        occ[y0:y1, x0:x1] = self.grid[y0:y1, x0:x1] > CONSIDERED_OCCUPIED

        cells = astar(occ, start, goal)
        if cells is None:
            self.get_logger().warn("No possible trajectory could be found")
            return

        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()

        for cell in cells:
            x, y = cell_to_world(cell[0], cell[1], self.map_origin_x, self.map_origin_y, self.map_res)
            p = PoseStamped()
            p.header = path.header
            p.pose.position.x = x
            p.pose.position.y = y
            path.poses.append(p)

        self.pub_path.publish(path)
            
    
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