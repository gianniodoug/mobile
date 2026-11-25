import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid

RESOL = 0.05
ORIGIN_X = -1.5
ORIGIN_Y = -2.0
WIDTH = 110
HEIGHT = 80
WALL_R = 0.15
SEGMENTS = [
    (1.0, 0.8, 2.0, 0.8),
    (1.0, -0.8, 2.0, -0.8),
    (2.0, 0.8, 2.0, -0.8),
]


def seg_dist(x, y, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    t = ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(x - (ax + t * dx), y - (ay + t * dy))


class Fake_Map(Node):
    def __init__(self):
        super().__init__('fake_map')
        map_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.map_pub = self.create_publisher(OccupancyGrid, '/robot/map', map_qos)

        iy, ix = np.mgrid[0:HEIGHT, 0:WIDTH]
        x = (ix + 0.5) * RESOL + ORIGIN_X
        y = (iy + 0.5) * RESOL + ORIGIN_Y
        grid = np.zeros((HEIGHT, WIDTH), dtype=np.int8)
        for ax, ay, bx, by in SEGMENTS:
            grid[seg_dist(x, y, ax, ay, bx, by) < WALL_R] = 100

        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = RESOL
        msg.info.width = WIDTH
        msg.info.height = HEIGHT
        msg.info.origin.position.x = ORIGIN_X
        msg.info.origin.position.y = ORIGIN_Y
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.flatten().tolist()
        self.map_pub.publish(msg)
        self.get_logger().info(f'Fake map published: {int((grid == 100).sum())} occupied cells')


def main(args=None):
    rclpy.init(args=args)
    node = Fake_Map()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()