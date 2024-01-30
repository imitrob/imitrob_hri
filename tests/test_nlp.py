
import rclpy, time, ast
from rdflib.term import URIRef
import numpy as np
from numpy import array
try:
    from imitrob_hri.imitrob_nlp.sentence_processor_node import SentenceProcessor
    from crow_msgs.msg import SentenceProgram
    from imitrob_hri.imitrob_nlp.modules.ObjectDetector import ObjectDetector
    from imitrob_hri.imitrob_nlp.modules.GrammarParser import GrammarParser
    from imitrob_hri.imitrob_nlp.database.Ontology import RobotProgram, RobotProgramOperator, RobotProgramOperand, RobotCustomProgram, Template
    from imitrob_hri.imitrob_nlp.structures.tagging.ParsedText import ParseTreeNode
    from crow_ontology.crowracle_client import CrowtologyClient

except Exception as e:
    SentenceProcessor = None
    SentenceProcessor_e = e



def test_failure():
    time.sleep(6)

def test_import():
    ''' Require the Ontology and parameter server running '''
    assert SentenceProcessor is not None, SentenceProcessor_e
    rclpy.init()
    sp = SentenceProcessor()
    rclpy.spin_once(sp)
    sp.destroy_node()
    
    rclpy.shutdown()
    
''' Notes:
FAILED tests/test_nlp.py::test_nlp_1 - KeyError: 'processor_busy_flag'

Object specified MUST be visible in Ontology
'''
def test_nlp_1():
    # particular reason why it is here 
    from imitrob_hri.imitrob_nlp.sentence_processor_node import SentenceProcessor
    try:
        rclpy.init()
    except RuntimeError:
        pass
    sp = SentenceProcessor()

    # print([f"{oo['uri']}: {oo['color_nlp_name_CZ']}\n" for oo in sp.crowracle.getTangibleObjects()])
    msg = SentenceProgram()
    
    # semantically must match the wanted object
    example_list = [
        # Success (cube_holes visible)
        ["Seber červenou kostku", {'target_action': 'pick', 'target_object': 'cube_holes', 'to_color': 'red'}],
        ["Ukaž na červenou kostku", {'target_action': 'point', 'target_object': 'cube_holes', 'to_color': 'red'}],
        ["Podej mi červenou kostku", {'target_action': 'pass', 'target_object': 'cube_holes', 'to_color': 'red'}],
        # Success (cube_holes visible)
        ["Seber kostku", {'target_action': 'pick', 'target_object': 'cube_holes'}],
        ["Ukaž na kostku", {'target_action': 'point', 'target_object': 'cube_holes'}],
        ["Podej mi kostku", {'target_action': 'pass', 'target_object': 'cube_holes'}],
        # Success (wheel visible)
        ["Seber kolo", {'target_action': 'pick', 'target_object': 'wheel'}],
        ["Ukaž na kolo", {'target_action': 'point', 'target_object': 'wheel'}],
        ["Podej mi kolo", {'target_action': 'pass', 'target_object': 'wheel'}],
        # Success (wafer visible)
        # ["Seber destičku", ('pick', 'wafer')],
        # ["Ukaž na destičku", ('point', 'wafer')],
        # ["Podej mi destičku", ('pass', 'wafer')],
        # Success (hammer NOT visible)
        ["Seber kladivo", {'target_action': 'Noop'}],
        ["Ukaž na kladivo", {'target_action': 'Noop'}],
        ["Podej mi kladivo",{'target_action': 'Noop'}],
        # Testing some synonyms
        ["Zvedni kostku", {'target_action': 'pick', 'target_object': 'cube_holes'}],
        ["Získej kostku", {'target_action': 'pick', 'target_object': 'cube_holes'}],
        ["Dej mi kostku", {'target_action': 'pass', 'target_object': 'cube_holes'}],
        ["Dej mi krychli",{'target_action': 'pass', 'target_object': 'cube_holes'}],
        # Shuffled
        ["Dej mi krychli s dírama.", {'target_action': 'pass', 'target_object': 'cube_holes'}],
        ["Kostku mi dej.", {'target_action': 'pass', 'target_object': 'cube_holes'}],
        ["dej kostku", {'target_action': 'pass', 'target_object': 'cube_holes'}],
        # Program template has two actions 
        ["Kostku mi dej. A pak ukaž na kostku.", {'target_action': 'pass', 'target_object': 'cube_holes'}],
        # TODO: Barvy, cannot assign the color to object in order to test this
        # ["Podej mi zelenou kostku.", {'target_action': 'pass', 'target_object': 'cube_holes'}],
        # ["Podej mi červenou kostku.", {'target_action': 'pass', 'target_object': 'cube_holes'}],
        # ["Podej mi kladivo", ('Noop', '')],
        
        ["Podej mi červenou kostku", {'target_action': 'pass', 'target_object': 'cube_holes', 'to_color': 'red'}],
        ["Podej mi zelenou kostku", {'target_action': 'pass', 'target_object': 'cube_holes', 'to_color': 'green'}],
        
        ["Nalij kostku na kolo.", {'target_action': 'pour', 'target_object': 'cube_holes', 'target_storage': 'wheel'}]
        # ["Ukaž na modrý kolík", ('point', 'blue peg')],
        # ["Ukaž na zelenou kostku", ('point', 'green cube')],
        # ["Ukaž na kolík", ('', '')],
        # ["Ukaž na zelenou kostku", ('', '')],
        # ["Definuj modrou pozici", ('', '')],
        # ["Pick a red cube", ('', '')],
        # ["Seber kostku", ('', '')],
    ]
    '''example_list = [    
        ["Dej zelenou kostku do červený krabice.", ('place', 'cube_holes', 'paper box')],
        # FUTURE?
        # ["Vezmi jablko a hrušku a dej je do krabice", (..)]
        ["Dej zelenou kostku na červenou kostku"]
    ]'''
    
    for sentence, solution in example_list:
        print("===========================================================")
        print("======================= NEW  SAMPLE =======================")
        print("===========================================================")
        msg.header.stamp = sp.get_clock().now().to_msg()
        msg.data = [sentence]
            
        out = sp.process_sentence_callback(msg, out=True)
        t = ast.literal_eval(out.data[0])

        # target_object must match == name must match == (cube_od_1 == cube_od_1)
        # exact matching: somewhere the we get objects on the scene, we choose some object
        # and compare this specific object with the object from the sentence
        # common mapping: we get definition of the properties that must match

        def target_object_struri_to_type(uri):
            return URIRef(uri).fragment.split("_od_")[0]

        if 'target_object' in t.keys():
            t['target_object'] = target_object_struri_to_type(t['target_object'])
        if 'target_storage' in t.keys():
            t['target_storage'] = target_object_struri_to_type(t['target_storage'])


        for key in solution.keys():
            # HRICommand key must be equal to true value
            assert t[key] == solution[key], f"sentence: {sentence}\n{key}: {t[key]} != {solution[key]}\nraw solution: {t}"
        # input("next?")
    
    sp.destroy_node()
    rclpy.shutdown()
    print("Success")


def test_target_objects_detector():
        # particular reason why it is here 
    from imitrob_hri.imitrob_nlp.sentence_processor_node import SentenceProcessor
    try:
        rclpy.init()
    except RuntimeError:
        pass
    sp = SentenceProcessor()
    msg = SentenceProgram()
    
    # semantically must match the wanted object
    example_list = [
        # Success (cube_holes visible)
        ["Seber červenou kostku", "kostka", True],
        ["Seber červenou kostku na kolo", "kolo", False],
        ["Seber červenou kostku na kolo", "kostka", True],
        ["Na kolo Seber červenou kostku", "kolo", False],
        ["Na kolo Seber červenou kostku", "kostka", True],
        ["Seber hustou červenou kostku pod listí.", "kostka", True],
        ["Seber hustou červenou kostku pod listí na lavici.", "kostka", True],
        ["Pod lavicí seber hustou červenou kostku pod listí.", "kostka", True],
        ["Pod kolem seber hustou červenou kostku pod listí.", "kostka", True],
        ["Pod kolem seber hustou červenou kostku pod listí.", "kolo", True],
    ]
    
    for sentence, to, solution in example_list:
        msg.header.stamp = sp.get_clock().now().to_msg()
        msg.data = [sentence]
            
        out = test_target_objects_detector_inner(sp, msg, to, is_target_object_true=solution)
        
        # t = ast.literal_eval(out.data[0])
        # target_object must match == name must match == (cube_od_1 == cube_od_1)
        # exact matching: somewhere the we get objects on the scene, we choose some object
        # and compare this specific object with the object from the sentence
        # common mapping: we get definition of the properties that must match

    
    sp.destroy_node()
    rclpy.shutdown()
    print("Success")


def test_target_objects_detector_inner(self, nlInputMsg, to, is_target_object_true):
            language = 'cs'

            self.pclient.processor_busy_flag = True
            self.pclient.nlp_ongoing = True

            input_sentences = nlInputMsg.data
            
            assert len(input_sentences) == 1, "TODO: len(input_sentences) > 1"
            for input_sentence in input_sentences:
                input_sentence = self.replace_synonyms(input_sentence)
                input_sentence = input_sentence.lower()

                # template_speech_sorted = self.nl_processor.process_text(input_sentence)
                gp = GrammarParser(language = language)
                parsed_text = gp.parse(input_sentence)
                root = parsed_text.parse_tree

                program = RobotProgram()
                program.root = RobotProgramOperator(operator_type="AND")
                for subnode in root.subnodes:
                    if type(subnode) is ParseTreeNode:
                        # create a single robot instructon
                        tagged_text = subnode.flatten()

                        od = ObjectDetector(language = language, client = CrowtologyClient(node=self))
                        # objs_det = od.detect_object(tagged_text)

                        print("0  ", to, tagged_text)
                        is_target_object = od.is_target_object(to, tagged_text)
                        print("1  ", is_target_object_true, is_target_object)
                        assert is_target_object_true == is_target_object, "Failure"
                        
# test_target_objects_detector()
test_nlp_1()
    
def test_str_dict_conversion():
    
    test_str = '{"target_action": "pick", "target_object": [], \
                "target_object_probs": [], "actions": ["pick"], \
                "action_probs": [1.0], "action_timestamp": 0.0, \
                "objects": [[]], "object_probs": [], \
                "object_classes": ["object"], "parameters": "", \
                "objs_mentioned_cls": "cube", \
                "objs_mentioned_cls_probs": [0., 1., 0., 0.], \
                "objs_mentioned_properties": []}'
    
    test_dict = ast.literal_eval(test_str)

    test_str_ = str(test_dict)
    
    test_dict_ = ast.literal_eval(test_str_)
    
    assert test_dict_ == test_dict
    
    