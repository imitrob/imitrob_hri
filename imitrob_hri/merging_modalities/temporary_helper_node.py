import sys, os, time
import rclpy
from rclpy.node import Node
import numpy as np

# from std_msgs.msg import Int8, Float64MultiArray, Int32, Bool, MultiArrayDimension, String
# from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, Vector3
# from moveit_msgs.msg import RobotTrajectory
# from trajectory_msgs.msg import JointTrajectoryPoint
# from teleop_msgs.msg import EEPoseGoals, JointAngles
# from visualization_msgs.msg import MarkerArray, Marker
# from sensor_msgs.msg import JointState

from teleop_msgs.msg import HRICommand
from std_msgs.msg import Bool
# Something like:
# from teleop_msgs.srv import MergeModalities
from modality_merger import ModalityMerger, ProbsVector
from configuration import Configuration3

class TemporaryHelperNode(Node):
    def __init__(self):
        """Standard ROS Node
        """        
        super().__init__('temporary_helper_node')


        self.mm_publisher = self.create_publisher(HRICommand, '/mm/solution', 5)

        

rclpy.init()
service = Service()
rclpy.spin(service) # Needed to serve (? - verify)
