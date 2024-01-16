
import rclpy, time, ast
import numpy as np
from numpy import array
try:
    from imitrob_hri.imitrob_nlp.sentence_processor_node import SentenceProcessor
    from crow_msgs.msg import SentenceProgram
    
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


'''
def test_nlp_1():
    try:
        rclpy.init()
    except RuntimeError:
        pass
    sp = SentenceProcessor()
    msg = SentenceProgram()
    
    example_list = [
        ["Seber červenou kostku", ('pick', 'cube')],
        # ["Ukaž na červenou kostku", ('point', 'cube')],
        # ["Podej mi červenou kostku", ('pass', 'cube')],
        #
        # ["Ukaž na modrý kolík", ('point', 'blue peg')],
        # ["Ukaž na zelenou kostku", ('point', 'green cube')],
        # ["Ukaž na kolík", ('', '')],
        # ["Ukaž na zelenou kostku", ('', '')],
        # ["Definuj modrou pozici", ('', '')],
        # ["Pick a red cube", ('', '')],
        # ["Seber kostku", ('', '')],
    ]
    
    for sentence, solution in example_list:
        msg.header.stamp = sp.get_clock().now().to_msg()
        msg.data = [sentence]
            
        out = sp.process_sentence_callback(msg, out=True)
        print("PROSTENECO: ", out.data[0])
        t = ast.literal_eval(out.data[0])
        assert t['target_action'] == solution[0], f"target_action: {t['target_action']} != {solution[0]}"
        assert t['target_object'] == solution[1], f"target_object: {t['target_object']} != {solution[1]}"
    
    sp.destroy_node()
    rclpy.shutdown()

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
    
    