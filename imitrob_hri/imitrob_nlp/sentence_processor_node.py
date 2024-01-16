#!/usr/bin/env python3
"""
Copyright (c) 2023 CIIRC, CTU in Prague
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.

@author: Karla Štěpánová, Megi Mejdrechová
@mail:  karla.stepanova@cvut.cz
"""
import json
import re

import rclpy
from rcl_interfaces.msg import ParameterType, ParameterDescriptor
from crow_msgs.msg import GestureSentence
from std_msgs.msg import Bool
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from crow_msgs.msg import SentenceProgram, StampedString # , ProcessedSentence, StringList, CommandType, #, NlpStatus

from imitrob_hri.imitrob_nlp.processing.NLProcessor import NLProcessor
import traceback as tb
import time
import numpy as np
from crow_ontology.crowracle_client import CrowtologyClient
from rdflib.namespace import Namespace, RDF, RDFS, OWL, FOAF, XSD
from imitrob_hri.imitrob_nlp.processing.ProgramRunner import ProgramRunner
from imitrob_hri.imitrob_nlp.modules.UserInputManager import UserInputManager
from crow_params.client import ParamClient

ONTO_IRI = "http://imitrob.ciirc.cvut.cz/ontologies/crow"
CROW = Namespace(f"{ONTO_IRI}#")

NOT_PROFILING = True
if not NOT_PROFILING:
    from crow_control.utils.profiling import StatTimer

from teleop_msgs.msg import HRICommand

# def distance(entry):
#     return entry[-1]


class SentenceProcessor(Node):
    #CLIENT = None
    MAX_OBJ_REQUEST_TIME = 2.5

    def __init__(self, node_name="sentence_processor"):
        super().__init__(node_name)
        self.DEBUG_MODE = False

        self.crowracle = CrowtologyClient(node=self)
        #CLIENT = self.crowracle
        self.onto = self.crowracle.onto
        self.LANG = 'cs'
        # self.get_logger().info(self.onto)
        self.ui = UserInputManager(language = self.LANG)
        self.guidance_file = self.ui.load_file('guidance_dialogue.json')
        self.templ_det = self.ui.load_file('templates_detection.json')
        self.synonyms_file = self.ui.load_file('synonyms.json')
        
        self.pclient = ParamClient()
        self.pclient.define("processor_busy_flag", False) # State of the sentence processor
        self.pclient.define("halt_nlp", False) # If true, NLP input should be halted
        self.pclient.define("nlp_ongoing", False)#ongoing speech and its processing
        self.pclient.define("silent_mode", 1) # Set by the user (level of the talking - 1 Silence, 2 - Standard speech, 3 - Debug mode/Full speech)
        self.pclient.define("ready_for_next_sentence", True) # If true, sentence processor can process and send next command
        self.pclient.define("can_start_talking", True) # If true, can start playing buffered sentences
        self.pclient.define("det_obj", "-")
        self.pclient.define("det_command", "-")
        self.pclient.define("det_obj_name", "-")
        self.pclient.define("det_obj_in_ws", "-")
        self.pclient.define("status", "-")
        self.pclient.define("nlp_alive", True)

        #create listeners (synchronized)
        self.nlTopic = "/nl_input"
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        listener = self.create_subscription(msg_type=SentenceProgram,
                                          topic=self.nlTopic,
                                          callback=self.process_sentence_callback,qos_profile=qos) #the listener QoS has to be =1, "keep last only".
        self.sentence_publisher = self.create_publisher(StampedString, "/nlp/command", qos)
        self.create_timer(0.3, self.keep_alive)

        self.nl_processor = NLProcessor(language = self.LANG, client = self.crowracle)
        self.get_logger().info('Input listener created on topic: "%s"' % self.nlTopic)
        if not NOT_PROFILING:
            StatTimer.init()

        self.sentence_publisher_hri_command = self.create_publisher(HRICommand, "/nlp/hri_command", qos)
        print("[NLP] Ready")

    def keep_alive(self):
        self.pclient.nlp_alive = time.time()

    def send_status(self, status="zadejte pozadavek", template_type="-", object_detected="-", found_in_ws=False):
        self.pclient.det_obj = object_detected
        self.pclient.det_command = template_type
        #self.pclient.det_obj_name = ??? # use crowtology client function, see visualizator
        self.pclient.det_obj_in_ws = found_in_ws # change to string 'ano', 'ne'?, see visualizator
        self.pclient.status = status

    def process_sentence_callback(self, nlInputMsg, out=False):
        ''' Main function '''
        print("process_sentence_callback")
        if not NOT_PROFILING:
            StatTimer.enter("semantic text processing")
        self.pclient.processor_busy_flag = True
        self.pclient.nlp_ongoing = True

        print(nlInputMsg)
        input_sentences = nlInputMsg.data
        print('I got sentence')

        data = []
        
        assert len(input_sentences) == 1, "TODO: len(input_sentences) > 1"
        for input_sentence in input_sentences:
            #self.get_database(write=True)
            input_sentence = self.replace_synonyms(input_sentence)
            input_sentence = input_sentence.lower()

            if True:
                goto_next_command = False
                found_in_ws = False
                success = False

                #process input sentence
                program_template_speech = self.nl_processor.process_text(input_sentence)
                # get current database state for writing an ungrounded and currently grounded program to be executed
                print("--------")
                print("Program Template:")
                print(program_template_speech)
                
                hricommand = self.template_to_hricommand(program_template_speech.root.children[0].template)
                
                self.sentence_publisher_hri_command.publish(hricommand)
                if out:
                    return hricommand
                else:
                    return

    def replace_synonyms(self, command):
        """place words in command with synonyms defined in synonym_file"""
        # root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
        # synonyms_filename = os.path.join(
        #     root_dir, "utils", "synonyms.json")
        synonyms = self.synonyms_file
        for key in synonyms[self.LANG]:
            for phrase in synonyms[self.LANG][key]:
                if phrase in command.lower():
                    src_str = re.compile(r'\b{}\b'.format(phrase), re.IGNORECASE)
                    command = src_str.sub(key, command)
        command = self.strip_extra_spaces(re.sub(r"\stodelete", "", command))
        number_words = re.findall(r'\b({}|{})\b\s*[0-9]'.format("číslo", "number"), command, re.IGNORECASE)
        for x in number_words:
            command = re.sub(r'\b{}\b'.format(x), "", command, re.IGNORECASE)
        return command

    def strip_extra_spaces(self, text):
        stripped_spaces = re.sub(' +', ' ', text)
        stripped_text = stripped_spaces.strip()
        return stripped_text

    def template_to_hricommand(self, template):
        
        d = {}
        d['target_action'] = template.target_action
        d['target_object'] = template.target_object 
        d['target_object_probs'] = list(template.target_object_probs)
        d['actions'] = [template.target_action]
        d['action_probs'] = [1.0]
        d['action_timestamp'] = 0.0
        d['objects'] = template.target_object
        d['object_probs'] = list(template.target_object_probs)
        d['object_classes'] = ['object']
        d['parameters'] = ""        
        
        d["objs_mentioned_cls"] = template.objs_mentioned_cls
        d["objs_mentioned_cls_probs"] = list(template.objs_mentioned_cls_probs)
        d["objs_mentioned_properties"] = template.objs_properties
        
        return HRICommand(data=[str(d)]) 
        
    # def make_one_hot_vector(self, template_names, activated_template_name):
    #     ''' For Language likelihoods vector construction
    #     '''
    #     a = np.zeros(len(template_names))
    #     n = template_names.index(activated_template_name)
    #     a[n] = 1.
    #     return a
    
    # def get_language_templates(self):
    #     return self.nl_processor.tf.get_all_template_names()
    
    # def get_language_detected_selection(self, program_template):
    #     template = program_template.root.children[0].template
    #     if template is None:
    #         raise AttributeError("Unknown command: {}".format(program_template))
    #     if not template.is_filled():
    #         # self.wait_then_talk()
    #         # print(f"Delay at the end: {time.time() - start_attempt_time}")
    #         return

    #     dict1 = template.get_inputs()
    #     template_type = dict1.get('action_type', '-')

    #     target_obj = dict1.get('target')
    #     if target_obj is not None:
    #         if hasattr(target_obj, "flags") and 'last_mentioned' in target_obj.flags:
    #         #TODO add last mentioned correferenced object and then delete this and adjust in ObjectGrounder
    #             target_obj = None
    #             dict1['target']=None
    #     if target_obj is not None:
    #         if len(target_obj.is_a) > 1:
    #                 # self.wait_then_talk()
    #                 print('TODO ask for specification which object to choose')
    #             #@TODO: ask for specification, which target_obj to choose
    #         target_obj = target_obj.is_a[0].n3()

    #         # Select only the name
    #         # '<http://imitrob.ciirc.cvut.cz/ontologies/crow#Cube>' -> 'Cube'
    #         return target_obj[1:-1].split("#")[-1]            
    #     else:
    #         return None

    # def wait_for_can_start_talking(self):
    #     if self.pclient.silent_mode is None:
    #         self.pclient.define("silent_mode", 1)
    #     if self.pclient.silent_mode > 1:
    #         while self.pclient.can_start_talking == False:
    #             time.sleep(0.2)
    #         self.pclient.can_start_talking = False

    # def wait_then_talk(self):
    #     if self.pclient.silent_mode is None:
    #         self.pclient.define("silent_mode", 1) # Set by the user (level of the talking - 1 Silence, 2 - Standard speech, 3 - Debug mode/Full speech)
    #     if self.pclient.silent_mode > 1:
    #         while self.pclient.can_start_talking == False:
    #             time.sleep(0.2)
    #         self.pclient.can_start_talking = False
    #     self.ui.buffered_say(flush=True, level=self.pclient.silent_mode)
    #     self.pclient.can_start_talking = True

    # def run_program(self, program_template):
    #     program_runner = ProgramRunner(language = self.LANG, client = self.crowracle)
    #     robot_program = program_runner.evaluate(program_template)
    #     print()
    #     print("Grounded Program")
    #     print("--------")
    #     print(robot_program)
    #     return robot_program


def main():
    rclpy.init()
    try:
        sp = SentenceProcessor()
        rclpy.spin(sp)
        sp.destroy_node()
    except KeyboardInterrupt:
        print("User requested shutdown.")
    except BaseException as e:
        print(f"Some error had occured: {e}")
        tb.print_exc()

if __name__ == "__main__":
    main()