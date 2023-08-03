#!/usr/bin/env python3
"""
Copyright (c) 2023 CIIRC, CTU in Prague
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.

@author: Karla Štěpánová, Megi Mejdrechová
@mail:  karla.stepanova@cvut.cz
"""
# if __name__ == '__main__':
#     import sys
#     sys.path.append("/home/petr/crow-base/src/crow_nlp/crow_nlp")
#     sys.path.append("/home/petr/crow-base/src/crow_ontology")
#     sys.path.append("/home/petr/crow-base/src/crow_utils")
#     sys.path.append("/home/petr/crow-base/src/crow_params")


import json
import re

import rclpy
from rcl_interfaces.msg import ParameterType, ParameterDescriptor
from crow_msgs.msg import GestureSentence
from std_msgs.msg import Bool
from rclpy.node import Node
# from ros2param.api import call_get_parameters
# from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from crow_msgs.msg import SentenceProgram, StampedString # , ProcessedSentence, StringList, CommandType, #, NlpStatus

from crow_nlp.nlp_crow.processing.NLProcessor import NLProcessor
# import message_filters
import traceback as tb
# import curses
import time
import numpy as np
# from datetime import datetime
# from curses.textpad import Textbox, rectangle
from crow_ontology.crowracle_client import CrowtologyClient
from rdflib.namespace import Namespace, RDF, RDFS, OWL, FOAF, XSD
# from rdflib import URIRef, BNode, Literal, Graph
# from rdflib.term import Identifier
from crow_nlp.nlp_crow.processing.ProgramRunner import ProgramRunner
from crow_nlp.nlp_crow.modules.UserInputManager import UserInputManager
from crow_params.client import ParamClient

ONTO_IRI = "http://imitrob.ciirc.cvut.cz/ontologies/crow"
CROW = Namespace(f"{ONTO_IRI}#")

NOT_PROFILING = True
if not NOT_PROFILING:
    from crow_control.utils.profiling import StatTimer

import utils
import sys; sys.path.append("..")
from merging_modalities.modality_merger import ModalityMerger, UnifiedSentence

def distance(entry):
    return entry[-1]


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

        # Gestures
        self.create_subscription(msg_type=Bool, topic='/gesture/ongoing',
            callback=self.gestures_ongoing_callback,qos_profile=qos)
        self.gestures_ongoing_now = False
        self.create_subscription(msg_type=GestureSentence, topic='/gesture/gesture_sentence',
            callback=self.gesture_sentence_callback,qos_profile=qos)
        self.gesture_sentence = GestureSentence()

    def gestures_ongoing_callback(self, msg):
        ''' Updates current value of person gesturing bool
        '''
        self.gestures_ongoing_now = msg.data
    def gesture_sentence_callback(self, msg):
        self.gesture_sentence = msg

    def keep_alive(self):
        self.pclient.nlp_alive = time.time()

    def send_status(self, status="zadejte pozadavek", template_type="-", object_detected="-", found_in_ws=False):
        self.pclient.det_obj = object_detected
        self.pclient.det_command = template_type
        #self.pclient.det_obj_name = ??? # use crowtology client function, see visualizator
        self.pclient.det_obj_in_ws = found_in_ws # change to string 'ano', 'ne'?, see visualizator
        self.pclient.status = status

    def wait_for_gestures_to_end(self):
        if self.gestures_ongoing_now:
            # Waits to the end of gesturing
            i = 0
            while self.gestures_ongoing_now:
                time.sleep(0.01)
                i += 1
                if i > 100:
                    print("Still gesturing!")
                    i = 0
        return
    
    def mm_gather_info(self, language_template_name, language_object_name):
        # 1. Templates Preprocessing 
        language_template_name = utils.template_name_to_default_name(language_template_name)
        language_templates = [utils.template_name_to_default_name(name) for name in self.get_language_templates()]
        gesture_templates = self.gesture_sentence.actions
        # 2. Make template likelihoods
        language_template_likelihoods = self.make_language_likelihoods(language_templates, language_template_name)
        gesture_template_likelihoods = self.gesture_sentence.action_likelihoods
        # 3. Unify template vectors
        templates, t_g, t_l = self.unify(gesture_templates, language_templates, \
                                    gesture_template_likelihoods, language_template_likelihoods)

        # 4. Objects preprocessing
        language_object_name = language_object_name
        language_objects = [language_object_name]
        gesture_objects = self.gesture_sentence.objects
        # 5. Make object likelihoods
        language_object_likelihoods = self.make_language_likelihoods(language_objects, language_object_name)
        gesture_object_likelihoods = self.gesture_sentence.object_likelihoods
        # 6. Unify object vectors
        # TODO: GET ALL OBJECT NAMES FROM THE SCENE
        objects, o_g, o_l = self.unify(gesture_objects, language_objects, \
                                       gesture_object_likelihoods, language_object_likelihoods)

        # Checker
        print("[Modality merger] Prepare the gestures:")
        if (time.time() - self.gesture_sentence.header.stamp.sec) > 3.:
            if self.gesture_sentence.header.stamp.sec == 0:
                print("     [WARNING!] No gestures received")
            else:
                print(f"    [WARNING!] Gesture sentence old {(time.time() - self.gesture_sentence.header.stamp.sec)} sec.")

        print("REPORT")
        print(list(zip(templates, t_g, t_l)))
        print(list(zip(objects, o_g, o_l)))
        gs = UnifiedSentence(target_action=t_g, target_objects=o_g)
        ls = UnifiedSentence(target_action=t_l, target_objects=o_l)

        return (templates, objects), (gs, ls)
    
    def mm_run(self, mm_info):
        templates, objects = mm_info[0]
        gs, ls = mm_info[1]
        mm = ModalityMerger(templates, objects)
        print("[Modality merger] Here comes the magic:")
        todo = mm.feedforward(ls, gs)
        print("=====================================")
        print(todo)
        print("=====================================")

    @staticmethod
    def unify(gesture_templates, language_templates, gesture_likelihoods, language_likelihoods):
        ''' If language and gesture templates has different sizes or one/few templates are missing
            This function makes UNION from both template lists.
        '''
        print(f"[{len(gesture_templates)}] gesture_templates: {gesture_templates}") 
        print(f"[{len(language_templates)}] language_templates: {language_templates}")

        templates = gesture_templates.copy()
        templates.extend(language_templates)
        templates = list(set(templates))

        gesture_likelihoods_unified =  [0.] * len(templates)
        language_likelihoods_unified = [0.] * len(templates)
        for template in templates:
            if template in gesture_templates:
                n = gesture_templates.index(template)
                m = templates.index(template)
                gesture_likelihoods_unified[m] = gesture_likelihoods[n]

            if template in language_templates:
                n = language_templates.index(template)
                m = templates.index(template)
                language_likelihoods_unified[m] = language_likelihoods[n]
        
        print(f"[{len(templates)}] final templates: {templates}")
        return templates, language_likelihoods_unified, gesture_likelihoods_unified

    def make_language_likelihoods(self, template_names, activated_template_name):
        a = np.zeros(len(template_names))
        n = template_names.index(activated_template_name)
        a[n] = 1.
        return a
    
    def get_language_templates(self):
        return self.nl_processor.tf.get_all_template_names()

    def process_sentence_callback(self, nlInputMsg):
        # 1. Check if gestures are still in process of gesturing
        self.wait_for_gestures_to_end()
        # 2. Get processed gesture sentence
        # TODO: Check gesture sentence if is fresh
        

        if not NOT_PROFILING:
            StatTimer.enter("semantic text processing")
        self.pclient.processor_busy_flag = True

        print(nlInputMsg)
        input_sentences = nlInputMsg.data
        print('I got sentence')

        data = []
        
        assert len(input_sentences) == 1, "TODO: len(input_sentences) > 1"
        for input_sentence in input_sentences:
            #self.get_database(write=True)
            input_sentence = self.replace_synonyms(input_sentence)
            input_sentence = input_sentence.lower()
            #self.ontoC = self.db.onto.__enter__()

            #check if changing to silent mode
            if self.templ_det[self.LANG]['silent'] in input_sentence:
                self.wait_for_can_start_talking()
                self.ui.say(self.guidance_file[self.LANG]["change_to_silence"])
                self.ui.say(self.guidance_file[self.LANG]["change_back_talk"])
                self.pclient.can_start_talking = True
                self.pclient.silent_mode = 1
            elif self.templ_det[self.LANG]['talk'] in input_sentence:
                if self.templ_det[self.LANG]['all'] in input_sentence:
                    self.wait_for_can_start_talking()
                    self.ui.say(self.guidance_file[self.LANG]["change_to_talking_lot"])
                    self.ui.say(self.guidance_file[self.LANG]["change_back_talk"])
                    self.ui.say(self.guidance_file[self.LANG]["change_back_silence"])
                    self.pclient.can_start_talking = True
                    self.pclient.silent_mode = 3
                else:
                    self.wait_for_can_start_talking()
                    self.ui.say(self.guidance_file[self.LANG]["change_to_talking"])
                    self.ui.say(self.guidance_file[self.LANG]["change_back_silence"])
                    self.pclient.can_start_talking = True
                    self.pclient.silent_mode = 2
            else:
                goto_next_command = False
                found_in_ws = False
                success = False

                #process input sentence
                program_template = self.nl_processor.process_text(input_sentence)
                # get current database state for writing an ungrounded and currently grounded program to be executed
                print("--------")
                print("Program Template:")
                print(program_template)

                start_attempt_time = time.time()
                while (time.time() - start_attempt_time) < self.MAX_OBJ_REQUEST_TIME:
                    # print(f"Delay: {time.time() - start_attempt_time}")
                    robot_program = self.run_program(program_template)
                    try:
                        template = program_template.root.children[0].template
                        if template is None:
                            raise AttributeError("Unknown command: {}".format(program_template))
                        if not template.is_filled():
                            # self.wait_then_talk()
                            # print(f"Delay at the end: {time.time() - start_attempt_time}")
                            continue

                        dict1 = template.get_inputs()
                        template_type = dict1.get('action_type', '-')

                        target_obj = dict1.get('target')
                        if target_obj is not None:
                            if hasattr(target_obj, "flags") and 'last_mentioned' in target_obj.flags:
                            #TODO add last mentioned correferenced object and then delete this and adjust in ObjectGrounder
                                target_obj = None
                                dict1['target']=None
                        if target_obj is not None:
                            if len(target_obj) > 1:
                                    # self.wait_then_talk()
                                    print('TODO ask for specification which object to choose')
                                #@TODO: ask for specification, which target_obj to choose
                            object_detected = target_obj
                            found_in_ws = True
                        else:
                            object_detected = '-'
                            found_in_ws = False
                        success = True
                        break
                    except AttributeError as  e:
                        print('No template found error')
                        self.send_status("neznamy prikaz")
                        self.wait_then_talk()
                        goto_next_command = True
                        break
                    # print(f"Delay at the end: {time.time() - start_attempt_time}")

                if goto_next_command:
                    continue
                if not success:
                    if not found_in_ws and hasattr(template, "target_ph_cls"):
                        self.get_logger().warn(f'Object of class {template.target_ph_cls} not found in workspace, quitting.')
                    continue
                self.send_status("zpracovavam", template_type, object_detected, found_in_ws)

                # NEW HERE
                mm_data = self.mm_gather_info(template_type)
                self.mm_run(mm_data)
                # NEW END

                if dict1.get('action', False):
                    data.append(dict1)
                    actions = json.dumps(data)
                    msg = StampedString()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.data = actions
                    print(f'will publish {msg.data}')
                    self.sentence_publisher.publish(msg)
                    self.send_status("pozadavek odeslan", template_type, object_detected, found_in_ws)
                    self.pclient.processor_busy_flag = False
                    if not self.DEBUG_MODE:
                        self.pclient.ready_for_next_sentence = False
                    if not NOT_PROFILING:
                        StatTimer.exit("semantic text processing")
                    if self.LANG == 'en':
                        template_type_en = [k for k, v in self.templ_det['cs'].items() if v == template_type]
                        template_type = template_type_en[0]
                    if dict1.get("command_buffer", 'main') == 'meanwhile':
                        self.ui.buffered_say(self.guidance_file[self.LANG]["will_publish_meanwhile"] + template_type, say=2)
                    else:
                        self.ui.buffered_say(self.guidance_file[self.LANG]["will_publish"] + template_type, say=2)
                    #     self.ui.buffered_say(flush=True, say=False)
                    self.wait_then_talk()
                else:
                    self.send_status("neznamy")
                    self.wait_then_talk()
                print('found ' + str(found_in_ws))
                print('heard ' + input_sentence)
                print('templ ' + template_type)
                print(f'object {object_detected}')


            if not NOT_PROFILING:
                StatTimer.try_exit("semantic text processing")

        #     if self.db_api.get_state() == State.DEFAULT:
        #         # self.save_unground_program(program_template)
        #         self.send_database()
        #         self.get_database(write=True)
        #
        #        # self.ontoC = self.db.onto.__enter__()
        #         robot_program = self.run_program(program_template)
        #         if self.db_api.get_state() == State.DEFAULT:
        #             self.save_grounded_program(robot_program)
        #             self.send_database()
        #         elif self.db_api.get_state() != State.LEARN_FROM_INSTRUCTIONS:
        #             self.db_api.set_state(State.DEFAULT)
        #             self.send_database()
        #
        #     elif self.db_api.get_state() == State.LEARN_FROM_INSTRUCTIONS:
        #         self.save_new_template(program_template)
        #         self.send_database()
        #
        # # print list of programs
        # self.get_database(write=False)
        # all_custom_templates = self.db_api.get_custom_templates()
        # for custom_template in all_custom_templates:
        #     print(custom_template.name[1:])
        # all_programs = self.db.onto.search(type=self.db.onto.RobotProgram)
        # path = os.path.dirname(os.path.abspath(__file__)) + '/saved_updated_onto.owl'
        # for program in all_programs:
        #     print(program.name)
        # return
        #


        # remove busy flag
        self.pclient.processor_busy_flag = False

        if self.DEBUG_MODE:
            print('Robot vykonává danou akci.')
            # time.sleep(10)
            #TODO comment out
            # change robot done flag

            self.pclient.ready_for_next_sentence = True

        return

    def wait_for_can_start_talking(self):
        if self.pclient.silent_mode is None:
            self.pclient.define("silent_mode", 1)
        if self.pclient.silent_mode > 1:
            while self.pclient.can_start_talking == False:
                time.sleep(0.2)
            self.pclient.can_start_talking = False

    def wait_then_talk(self):
        if self.pclient.silent_mode is None:
            self.pclient.define("silent_mode", 1) # Set by the user (level of the talking - 1 Silence, 2 - Standard speech, 3 - Debug mode/Full speech)
        if self.pclient.silent_mode > 1:
            while self.pclient.can_start_talking == False:
                time.sleep(0.2)
            self.pclient.can_start_talking = False
        self.ui.buffered_say(flush=True, level=self.pclient.silent_mode)
        self.pclient.can_start_talking = True

    def run_program(self, program_template):
        program_runner = ProgramRunner(language = self.LANG, client = self.crowracle)
        robot_program = program_runner.evaluate(program_template)
        print()
        print("Grounded Program")
        print("--------")
        print(robot_program)
        return robot_program

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