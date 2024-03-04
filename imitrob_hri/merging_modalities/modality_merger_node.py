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
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

from teleop_msgs.msg import HRICommand
from std_msgs.msg import Bool
# Something like:
# from teleop_msgs.srv import MergeModalities
from imitrob_hri.merging_modalities.modality_merger import ModalityMerger, ProbsVector, MMSentence
from imitrob_hri.merging_modalities.configuration import ConfigurationCrow1

from imitrob_hri.data.scene3_def import Scene3, Object3

from imitrob_templates.small_ontology_scene_reader import SceneOntologyClient

# optional - for checking all configuration templates exists
from imitrob_hri.imitrob_nlp.TemplateFactory import create_template

from imitrob_hri.imitrob_nlp.nlp_utils import cc
import threading
from copy import deepcopy
from pathlib import Path
import argparse


class MMNode(Node):
    def __init__(self, max_time_delay = 15., model = 1, use_magic = 'entropy', c=ConfigurationCrow1(), experiment_args=None):
        """Standard ROS Node

        Args:
            max_time_delay (Float, optional): Exectution takes place when all modalities are received under max_time_delay constant. Defaults to 5..
        """        
        super().__init__('merge_modalities_node')
        self.mm_publisher = self.create_publisher(HRICommand, '/mm/solution', 5)

        self.max_time_delay = max_time_delay
        

        self.create_subscription(HRICommand, '/gg/hri_command', self.receiveHRIcommandG, 10)
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(HRICommand, '/nlp/hri_command', self.receiveHRIcommandL, qos_profile=qos)
        
        self.receivedHRIcommandG = None
        self.receivedHRIcommandGstamp = 0.
        
        self.receivedHRIcommandL = None
        self.receivedHRIcommandLstamp = 0.
        
        self.model = model
        self.use_magic = use_magic
        self.c = c
        
        self.soc = SceneOntologyClient(self)
        self.soc.add_dummy_cube()
        print(self.soc.get_scene3())

        self.experiment_args = experiment_args
        
        for template in self.c.mm_pars_names_dict['template']:
            assert create_template(template) is not None, f"Template {template} not exists!"
        
        print("Initialized")
        
    def clean_up(self):
        self.receivedHRIcommandG = None
        self.receivedHRIcommandGstamp = 0.
        
        self.receivedHRIcommandL = None
        self.receivedHRIcommandLstamp = 0.
        
    def receiveHRIcommandG(self, msg):
        self.receivedHRIcommandGstamp = time.perf_counter()
        self.receivedHRIcommandG = msg
        
    def receiveHRIcommandL(self, msg):
        self.receivedHRIcommandLstamp = time.perf_counter()
        self.receivedHRIcommandL = msg

    def execution_trigger(self, delay_run):
        """  Should check ongoing topics
        Returns:
            Bool: Execution can begin
        """        ''''''
        # Ongoing bool message frequency should be 10Hz
        # -> Wait 30ms (factor 5)
        time.sleep(0.05)
        # If not gesturing and not speaking -> execute
        if self.receivedHRIcommandG is not None and self.receivedHRIcommandL is not None:
            print(f"Execution both modalities ready!")
            return 'run'
        elif self.receivedHRIcommandG is not None or self.receivedHRIcommandL is not None:
            if delay_run == -1:
                return 'run delay'
        else:
            return 'no'
        
    def merge_modalities(self):
        """ Calls merge modalities function
        Args: 
            Internally saved HRICommand objects used
            self.receivedHRIcommandG
            self.receivedHRIcommandL
        Returns:
            HRICommand: Merged Command
        """        ''''''
        G_dict = {
            'action_probs': [], 
            'actions': [],
            'object_probs': [],
            'objects': []}
        L_dict = {
            'action_probs': [], 
            'actions': [],
            'object_probs': [],
            'objects': []}
        
        if self.receivedHRIcommandG is not None: print(f"receivedHRIcommandGStringToParse: {self.receivedHRIcommandG.data[0]}")
        if self.receivedHRIcommandL is not None: print(f"receivedHRIcommandLStringToParse: {self.receivedHRIcommandL.data[0]}")
        
        if self.receivedHRIcommandG is not None:
            receivedHRIcommandGStringToParse = self.receivedHRIcommandG.data[0]
            G_dict = json.loads(receivedHRIcommandGStringToParse)

        if self.receivedHRIcommandL is not None:
            receivedHRIcommandLStringToParse = self.receivedHRIcommandL.data[0]
            L_dict = json.loads(receivedHRIcommandLStringToParse)
        
        # G_dict['target_action']
        # G_dict['action_timestamp']
        # G_dict['object_classes']
        # G_dict['parameters']
        
        # In crow experiment, there are only one object of each kind
        # cube,cup,can,foam,crackers,bowl,drawer_socket        
        # - Discard _od_ number here
        if 'objects' in G_dict:
            for n,o in enumerate(deepcopy(G_dict['objects'])):
                if '_od_' in o:
                    G_dict['objects'][n] = o.split("_od_")[0]
            # for n,o in enumerate(deepcopy(L_dict['objects'])):
            #     if '_od_' in o:
            #         G_dict['objects'][n] = o.split("_od_")[0]
        if 'objects' in L_dict:
            for n,o in enumerate(deepcopy(L_dict['objects'])):
                if '_od_' in o:
                    L_dict['objects'][n] = o.split("_od_")[0]


        if 'storages' in G_dict:
            for n,o in enumerate(deepcopy(G_dict['storages'])):
                if '_od_' in o:
                    G_dict['storages'][n] = o.split("_od_")[0]

        if 'storages' in L_dict:
            for n,o in enumerate(deepcopy(L_dict['storages'])):
                if '_od_' in o:
                    L_dict['storages'][n] = o.split("_od_")[0]

        G = {}
        if 'action_probs' in G_dict:
            G['template'] = ProbsVector(G_dict['action_probs'], G_dict['actions'], self.c)
        else:
            G['template'] = ProbsVector([], [], self.c)
        if 'object_probs' in G_dict:
            G['selections'] = ProbsVector(G_dict['object_probs'], G_dict['objects'], self.c)
        else:
            G['selections'] = ProbsVector([], [], self.c)
        # if 'storage_probs' in G_dict:
        #     G['storages'] = ProbsVector(G_dict['storage_probs'], G_dict['storages'], self.c)
        
        if 'storage_probs' in G_dict:
            G['storages'] = ProbsVector(G_dict['storage_probs'], G_dict['storages'], self.c)
        else:
            G['storages'] = ProbsVector([], [], self.c)

        L = {}
        if 'action_probs' in L_dict:
            L['template'] = ProbsVector(L_dict['action_probs'], L_dict['actions'], self.c)
        else:
            L['template'] = ProbsVector([], [], self.c)

        if 'object_probs' in L_dict:
            L['selections'] = ProbsVector(L_dict['object_probs'], L_dict['objects'], self.c)
        else:
            L['selections'] = ProbsVector([], [], self.c)

        if 'storage_probs' in L_dict:
            L['storages'] = ProbsVector(L_dict['storage_probs'], L_dict['storages'], self.c)
        else:
            L['storages'] = ProbsVector([], [], self.c)

        # if 'storage_probs' in L_dict:
        #     L['storages'] = ProbsVector(L_dict['storage_probs'], L_dict['storages'], self.c)

        mms = MMSentence(L=L, G=G)

        
        scene = self.soc.get_scene3()
        
        
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
            use_magic: {self.use_magic}")
        

        # baseline
        mm = ModalityMerger(self.c, self.use_magic)
        mms.M, DEBUGdata = mm.feedforward3(mms.L, mms.G, scene=scene, epsilon=self.c.epsilon, gamma=self.c.gamma, alpha_penal=self.c.alpha_penal, model=0, use_magic='baseline')
        # M1
        mms1 = deepcopy(mms)
        mm = ModalityMerger(self.c, self.use_magic)
        mms1.M, DEBUGdata = mm.feedforward3(mms.L, mms.G, scene=scene, epsilon=self.c.epsilon, gamma=self.c.gamma, alpha_penal=self.c.alpha_penal, model=1, use_magic=self.use_magic)
        # M2
        mms2 = deepcopy(mms)
        mm = ModalityMerger(self.c, self.use_magic)
        mms2.M, DEBUGdata = mm.feedforward3(mms.L, mms.G, scene=scene, epsilon=self.c.epsilon, gamma=self.c.gamma, alpha_penal=self.c.alpha_penal, model=2, use_magic=self.use_magic)
        # M3
        mms3 = deepcopy(mms)
        mm = ModalityMerger(self.c, self.use_magic)
        mms3.M, DEBUGdata = mm.feedforward3(mms.L, mms.G, scene=scene, epsilon=self.c.epsilon, gamma=self.c.gamma, alpha_penal=self.c.alpha_penal, model=3, use_magic=self.use_magic)

        
        # mms.merged_part_to_HRICommand()

        def float_array_to_str(x):
            x = list(x)
            substr = '['
            for x_ in x:
                substr += '"'+f'{x_:.5f}'+'",'
            substr = substr[:-1]
            substr += ']'
            return substr

        hricommandstrs = []
        for i, (model, name) in enumerate(zip([mms,mms1,mms2,mms3], ["baseline", "M1", "M2", "M3"])):
            final_s = '{' + f'"target_action": "{model.M["template"].activated}", "target_object": "{model.M["selections"].activated}", "target_storage": "{model.M["storages"].activated}",' 
            final_s+= f'"actions": {model.M["template"].names}, "action_probs": {float_array_to_str(model.M["template"].p)},'
            final_s+= f'"objects": {model.M["selections"].names} , "object_probs": {float_array_to_str(model.M["selections"].p)} , "object_classes": "TODO", '
            final_s+= f'"storages": {model.M["storages"].names} , "storage_probs": {float_array_to_str(model.M["storages"].p)} , '
            # final_s+= f'"parameters": "TODO", "action_timestamp": "TODO", "scene": {scene}'
            final_s+= f'"epsilon": "{self.c.epsilon}", "gamma": "{self.c.gamma}", "alpha_penal": "{self.c.alpha_penal}", "model index": "{i}", "use_magic": "{self.use_magic}",'
            final_s+= f'"merge_function": {self.use_magic}'
            final_s+= '}'
            final_s = final_s.replace("'", '"')

            # out_sentence = f'{name}: {model.M["template"].activated} {model.M["selections"].activated} {model.M["storages"].activated}'

            # print(f"final_s:   {str(final_s)}")
            hric = HRICommand(data=[final_s])
            print(f"{cc.H}=========== OUT: {name} ============={cc.E}")
            print(hric.data)
            sel_obj_name = model.M["selections"].activated
            if sel_obj_name is not None:
                try:
                    sel_obj = scene.get_object(sel_obj_name)
                except:
                    print(f"Object {sel_obj_name} not in Scene!")
                else:
                    print(f"Target obj props: {sel_obj.get_all_properties()}")
            else:
                print("No activated objects")
            print(f"{cc.H}============================={cc.E}")

            hricommandstrs.append(deepcopy(final_s))

        # Here load the subpath
        # subpath2 = Path(__file__).joinpath("..").joinpath("datareal") # cut python file and add
        subpath = "/home/imitlearn/crow-base/src/imitrob_hri/imitrob_hri/data_real"
        if self.experiment_args is None:
            exp_path = "/"
        else:
            exp_path = f"{self.experiment_args.experiment}/{self.experiment_args.alignment}/"
        # assert subpath == subpath2, f"subpath: {subpath} != {subpath2}"
        final_path = Path(os.path.join(subpath, exp_path))
        final_path.mkdir(parents=True, exist_ok=True)
        i = 0
        while True:
            if final_path.joinpath(f"{self.experiment_args.experiment}_{self.experiment_args.alignment}_{self.experiment_args.run}_{self.experiment_args.subject}_mms_trial_{i}.npy").exists():
                i+=1
            else:
                break
        np.save(final_path.joinpath(f"{self.experiment_args.experiment}_{self.experiment_args.alignment}_{self.experiment_args.run}_{self.experiment_args.subject}_mms_trial_{i}"), [mms,mms1,mms2,mms3])
        np.save(final_path.joinpath(f"{self.experiment_args.experiment}_{self.experiment_args.alignment}_{self.experiment_args.run}_{self.experiment_args.subject}_str_trial_{i}"), hricommandstrs)


        return hric

    
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
                'types': ['liquid-container', 'object', 'container'],
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
                'types': ['liquid-container', 'object', 'container'],
                'glued': np.random.randint(0,2),
            }
            storages.append(Object3(observations))

        scene = Scene3(objects, storages, template_names=c.templates)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", choices=['arity', 'property'])
    parser.add_argument("--alignment", "-a", choices=['a', 'nal', 'nag', 'nol', 'nog'], help="a = aligned\nnal = not aligned action from language\nnag = not aligned action from gestures\nnol = not aligned object from language\nnog = not aligned object from gestures")
    parser.add_argument("--run", "-r", type=int, default=0)
    parser.add_argument("--subject", "-s", type=str, default="-")
    args, _ = parser.parse_known_args()

    rclpy.init()
    mmn = MMNode(experiment_args=args)

    def spinning_threadfn():
        while rclpy.ok():
            rclpy.spin_once(mmn)
            time.sleep(0.01)

    spinning_thread = threading.Thread(target=spinning_threadfn, args=(), daemon=True)
    spinning_thread.start()

    delay_run = -1
    while rclpy.ok():
        print("--- new round ---")
        time.sleep(1)
        et = mmn.execution_trigger(delay_run)
        if et == 'run':
            mmn.mm_publisher.publish(mmn.merge_modalities())
            mmn.clean_up()
            delay_run = -1
        elif et == 'run delay':
            delay_run = 40
        
        if delay_run > 0:
            delay_run -= 1
            if mmn.receivedHRIcommandG is not None and mmn.receivedHRIcommandL is not None:
                receivedsecondmodalityintheprocess = 'Both L&G ready'
            else: 
                receivedsecondmodalityintheprocess = ''
            print(f"Execution in {delay_run} {receivedsecondmodalityintheprocess}")

        if delay_run == 0:
            delay_run = -1

            mmn.mm_publisher.publish(mmn.merge_modalities())
            mmn.clean_up()
        

if __name__ == '__main__':
    main()