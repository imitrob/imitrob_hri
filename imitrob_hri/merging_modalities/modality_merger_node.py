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
from imitrob_hri.merging_modalities.modality_merger import ModalityMerger, ProbsVector, MMSentence
from imitrob_hri.merging_modalities.configuration import ConfigurationCrow1

from imitrob_hri.data.scene3_def import Scene3, Object3

from imitrob_templates.small_ontology_scene_reader import SceneOntologyClient

# optional - for checking all configuration templates exists
from imitrob_templates.small_template_factory import create_template

from imitrob_hri.imitrob_nlp.nlp_utils import cc

class MMNode(Node):
    def __init__(self, max_time_delay = 5., model = 1, use_magic = 'entropy', c=ConfigurationCrow1()):
        """Standard ROS Node

        Args:
            max_time_delay (Float, optional): Exectution takes place when all modalities are received under max_time_delay constant. Defaults to 5..
        """        
        super().__init__('merge_modalities_node')
        self.mm_publisher = self.create_publisher(HRICommand, '/mm/solution', 5)

        self.max_time_delay = max_time_delay
        
        self.create_subscription(HRICommand, '/hri/command', self.receiveHRIcommandG, 10)
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
        
        self.soc = SceneOntologyClient(self)
        self.soc.add_dummy_cube()
        print(self.soc.get_scene3())
        
        for template in self.c.mm_pars_names_dict['template']:
            assert create_template(template) is not None, f"Template {template} not exists!"
        
        print("Initialized")
        
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
            self.mm_publisher.publish(self.merge_modalities())

    def receiveHRIcommandL(self, msg):
        self.receivedHRIcommandLstamp = time.perf_counter()
        self.receivedHRIcommandL = msg
        
        if self.execution_trigger():
            self.mm_publisher.publish(self.merge_modalities())
        
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
        
        print("template_probs", template_probs)
        print("template_names", template_names)
        print("object_names", object_names)
        print("object_probs", object_probs)

        mms = MMSentence(G = {
            'template': ProbsVector(template_probs, template_names, self.c),
            'selections': ProbsVector(object_probs, object_names, self.c),
            'storages': ProbsVector([], [], self.c),
        },
        L = {
            'template': ProbsVector(template_probs, template_names, self.c),
            'selections': ProbsVector(object_probs, object_names, self.c),
            'storages': ProbsVector([], [], self.c),
        })
        
        scene = self.soc.get_scene3()
        
        mm = ModalityMerger(self.c, self.use_magic)
        # print(
        #     f" I dont understand {self.c.mm_pars_names_dict} \n\
        #     L: \n\
        #     template: {mms.L['template']}, \n\
        #     selections: {mms.L['selections']}, \n\
        #     storages: {mms.L['storages']}, \n\
        #     G: \n\
        #     template: {mms.G['template']}, \n\
        #     selections: {mms.G['selections']}, \n\
        #     storages: {mms.G['storages']}, \n\
        #     scene: {scene} \n\
        #     epsilon: {self.c.epsilon} \n\
        #     gamma: {self.c.gamma} \n\
        #     alpha_penal: {self.c.alpha_penal} \n\
        #     model: {self.model} \n\
        #     use_magic: {self.use_magic} \n\
        #     "
        # )
        # print("=== make_conjunction ===")

        print(f"{cc.H}============================={cc.E}")
        print(f"{mms.L['template']}\n{mms.L['selections']}")
        print(f"{cc.H}============================={cc.E}")
        print(f"{mms.G['template']}\n{mms.G['selections']}")
        print(f"{cc.H}============================={cc.E}")

        print(f"{cc.H}============================={cc.E}")
        print(f"{cc.H}====== MAKE CONJUNCTION ====={cc.E}")
        print(f"{cc.H}============================={cc.E}")


        mms.make_conjunction(self.c)
        print("=== make_conjunction ===")
        print(
            f"L: \n\
template: {mms.L['template']}, \n\
selections: {mms.L['selections']}, \n\
storages: {mms.L['storages']}, \n\
G: \n\
template: {mms.G['template']}, \n\
selections: {mms.G['selections']}, \n\
storages: {mms.G['storages']}, \n\
scene: {scene} \n\
epsilon: {self.c.epsilon} \n\
gamma: {self.c.gamma} \n\
alpha_penal: {self.c.alpha_penal} \n\
model: {self.model} \n\
use_magic: {self.use_magic} \n\
"
        )
        
        mms.M, DEBUGdata = mm.feedforward3(mms.L, mms.G, scene=scene, epsilon=self.c.epsilon, gamma=self.c.gamma, alpha_penal=self.c.alpha_penal, model=self.model, use_magic=self.use_magic)
        
        # mms.merged_part_to_HRICommand()
        return HRICommand(data=[f'"target_action": "{mms.M["template"].activated}", \
                                "target_object": "{mms.M["template"]}", \
                                "actions": {mms.M["template"].names}, \
                                "action_probs": {mms.M["template"].p}, \
                                "objects": {mms.M["selections"].names} ,\
                                "object_probs": {mms.M["selections"].p} ,\
                                "object_classes": "TODO", \
                                "parameters": "TODO", \
                                "action_timestamp": "TODO", \
                                "scene": {scene} \
                                "epsilon": {self.c.epsilon} \
                                "gamma": {self.c.gamma} \
                                "alpha_penal": {self.c.alpha_penal} \
                                "model": {self.model} \
                                "use_magic": {self.use_magic} '])
    
    
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