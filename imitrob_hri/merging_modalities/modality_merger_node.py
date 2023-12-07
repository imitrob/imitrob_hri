import sys, os, time, json
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

from imitrob_hri.merging_modalities.utils import Scene3, Object3

class MMNode(Node):
    def __init__(self, max_time_delay = 5., model = 3, use_magic = 'entropy', c=Configuration3()):
        """Standard ROS Node

        Args:
            max_time_delay (Float, optional): Exectution takes place when all modalities are received under max_time_delay constant. Defaults to 5..
        """        
        super().__init__('merge_modalities_node')
        self.mm_publisher = self.create_publisher(HRICommand, '/mm/solution', 5)

        self.max_time_delay = max_time_delay
        
        self.create_subscription(HRICommand, '/teleop_gesture_toolbox/action_sentence_mapped', self.receiveHRIcommandG, 10)
        self.create_subscription(HRICommand, '/mm/natural_input', self.receiveHRIcommandL, 10)
        
        self.create_subscription(Bool, '/mm/gestures_ongoing', self.ongoingG, 10)
        self.create_subscription(Bool, '/mm/natural_ongoing', self.ongoingL, 10)
        
        self.create_subscription(HRICommand, '/mm/crow_scene', self.receive_scene, 10)
        
        self.receivedHRIcommandG = None
        self.receivedHRIcommandGstamp = 0.
        
        self.receivedHRIcommandL = None
        self.receivedHRIcommandLstamp = 0.
        
        # We won't use service
        # self.srv = self.create_service(SrvType, '<topic>', self.receiveHRIcommand)
        
        self.model = model
        self.use_magic = use_magic
        self.c = c
        
        
        self.isOnGoingG = False
        self.isOnGoingL = False
        
    def ongoingG(self):
        pass
        # TODO
    def ongoingL(self):
        pass
        # TODO
        
    def receiveHRIcommandG(self, msg):
        self.receivedHRIcommandGstamp = time.perf_counter()
        self.receivedHRIcommandG = msg
        
        if self.execution_trigger():
            self.mm_publisher(self.merge_modalities())

    def receiveHRIcommandL(self, msg):
        self.receivedHRIcommandLstamp = time.perf_counter()
        self.receivedHRIcommandL = msg
        
        if self.execution_trigger():
            self.mm_publisher(self.merge_modalities())
        
    def execution_trigger(self):
        """  Should check ongoing topics
        Returns:
            Bool: Execution can begin
        """        ''''''
        # Ongoing bool message frequency should be 10Hz
        # -> Wait 30ms (factor 3)
        time.sleep(0.03)
        # If not gesturing and not speaking -> execute
        if not self.isOnGoingL and not self.isOnGoingG:
             return True
         
        return False
        
    def merge_modalities(self):
        """ Calls merge modalities function
        Args: 
            Internally saved HRICommand objects used
            self.receivedHRIcommandG
            self.receivedHRIcommandL
        Returns:
            HRICommand: Merged Command
        """        ''''''
        inputs = self.receivedHRIcommandG, self.receivedHRIcommandL
        
        print("GG:")
        print(inputs[0])
        
        receivedHRIcommandGStringToParse = self.receivedHRIcommandG.data[0]
        receivedHRIcommandG_parsed = json.loads(receivedHRIcommandGStringToParse)
        ['template_probs']
        
        template_names = receivedHRIcommandG_parsed['actions']
        receivedHRIcommandG_parsed['target_action']
        template_probs = receivedHRIcommandG_parsed['action_probs']
        receivedHRIcommandG_parsed['action_timestamp']
        object_names = receivedHRIcommandG_parsed['objects']
        object_probs = receivedHRIcommandG_parsed['object_probs']
        receivedHRIcommandG_parsed['object_classes']
        receivedHRIcommandG_parsed['parameters']
        
        
        G = {
            'template': ProbsVector(template_probs, template_names, self.c),
            'selections': ProbsVector(object_probs, object_names, self.c),
            'storages': ProbsVector([], [], self.c),
        }
        L = {
            'template': ProbsVector([], [], self.c),
            'selections': ProbsVector([], [], self.c),
            'storages': ProbsVector([], [], self.c),
        }
        
        mm = ModalityMerger(self.c, self.use_magic)
        M, DEBUGdata = mm.feedforward3(L, G, scene=sample['x_scene'], epsilon=self.epsilon, gamma=self.c.gamma, alpha_penal=self.c.alpha_penal, model=self.model, use_magic=self.use_magic)
        
        return HRICommand(inputs)
    
    def receive_scene(self, scene):
        ''' Node sends crow ontology scene info to ROS topic. Here, the topic msg is received and 
            converted to MM Scene3 object
        '''
        
        for object_name in object_chosen_names:
            assert isinstance(object_name, str), f"object name is not string {object_name}"
            observations = {
                'name': object_name, 
                'size': np.random.random() * 0.5, # [m]
                'position': [np.random.random(), np.random.random(), 0.0], # [m,m,m]
                'roundness-top': np.random.random() * 0.2, # [normalized belief rate]
                'weight': np.random.random() * 4, # [kg]
                'contains': np.random.random(), # normalized rate being full 
                'contain_item': np.random.randint(0,2), # normalized rate being full 
                'types': ['liquid container', 'object'],
                'glued': np.random.randint(0,2),
            }
            objects.append(Object3(observations))

        storage_chosen_names = list(np.random.choice(storage_name_list, size=nstgs, replace=False))
        
        for storage_name in storage_chosen_names:
            assert isinstance(storage_name, str), f"object name is not string {storage_name}"
            observations = {
                'name': storage_name, 
                'size': np.random.random() * 0.5, # [m]
                'position': [np.random.random(), np.random.random(), 0.0], # [m,m,m]
                'roundness-top': np.random.random() * 0.2, # [normalized belief rate]
                'weight': np.random.random() * 4, # [kg]
                'contains': np.random.random(), # normalized rate being full 
                'contain_item': np.random.randint(0,2), # normalized rate being full 
                'types': ['liquid container', 'object'],
                'glued': np.random.randint(0,2),
            }
            storages.append(Object3(observations))

        scene = Scene3(objects, storages, template_names=c.templates)

def main():
    rclpy.init()
    mmn = MMNode()
    rclpy.spin(mmn)

if __name__ == '__main__':
    main()