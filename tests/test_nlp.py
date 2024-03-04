
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

def test_sourced_ros_environement():
    import bt_msgs.msg
    
''' Notes:
FAILED tests/test_nlp.py::test_nlp_1 - KeyError: 'processor_busy_flag'

Object specified MUST be visible in Ontology
'''
CUBE = 'cube'
# CUBE = 'cube_holes'
WHEEL = 'crackers'
# WHEEL = 'wheel'

class BaseSet():
    def _get_set_names_(self):
        ret = []
        for i in dir(self):
            if i[0] != '_':
                ret.append(i)
        return ret
    
    def __call__(self):
        sets_names = self._get_set_names_()    
        sets = {}
        for set_name in sets_names:
            sets[set_name] = getattr(self, set_name)
        return sets

    def __str__(self):
        sets = self()
        ret_string = ''
        for set_name in sets.keys():
            set = sets[set_name]
            ret_string += f'#### Test Category: {set_name}\n\n'
            for test in set:
                ret_string += f'**{test[0]}**\n    {test[1]}\n'
        return ret_string

class TestSet(BaseSet):
    test_object_and_color = [
    ["Seber červenou kostku", {'target_action': 'pick', 'target_object': CUBE, 'to_color': 'red'}],
    ["Ukaž na červenou kostku", {'target_action': 'point', 'target_object': CUBE, 'to_color': 'red'}],
    ["Podej mi červenou kostku", {'target_action': 'pass', 'target_object': CUBE, 'to_color': 'red'}],
    ["Podej mi zelenou kostku.", {'target_action': 'pass', 'target_object': CUBE}],
    ["Ukaž na modrý šuplík", {'target_action': 'point', 'target_object': 'drawer', 'to_color': 'blue'}],
    ]
    test_object = [
    ["Seber kostku", {'target_action': 'pick', 'target_object': CUBE}],
    ["Ukaž na kostku", {'target_action': 'point', 'target_object': CUBE}],
    ["Podej mi kostku", {'target_action': 'pass', 'target_object': CUBE}],
    ["Seber kolo", {'target_action': 'pick', 'target_object': WHEEL}],
    ["Ukaž na kolo", {'target_action': 'point', 'target_object': WHEEL}],
    ["Podej mi kolo", {'target_action': 'pass', 'target_object': WHEEL}],
    ]
    # ["Seber destičku", {'target_action': 'pick', 'target_object': 'wafer'}],
    # ["Ukaž na destičku", {'target_action': 'point', 'target_object': 'wafer'}],
    # ["Podej mi destičku", {'target_action': 'pass', 'target_object': 'wafer'}],
    # ]
    # test_unknown_object = [
    # ["Seber kladivo", {'target_action': 'Noop'}],
    # ["Ukaž na kladivo", {'target_action': 'Noop'}],
    # ["Podej mi kladivo",{'target_action': 'Noop'}],
    # ]
    test_synonyms = [
    ["Zvedni kostku", {'target_action': 'pick', 'target_object': CUBE}],
    ["Získej kostku", {'target_action': 'pick', 'target_object': CUBE}],
    ["Dej mi kostku", {'target_action': 'pass', 'target_object': CUBE}],
    ["Dej mi krychli",{'target_action': 'pass', 'target_object': CUBE}],
    ["Dej mi krychli s dírama.", {'target_action': 'pass', 'target_object': CUBE}],
    ]
    test_shuffled = [
    ["Kostku mi dej.", {'target_action': 'pass', 'target_object': CUBE}],
    ["dej kostku", {'target_action': 'pass', 'target_object': CUBE}],    
    ]
    test_multi_sentence = [
    ["Kostku mi dej. A pak ukaž na kostku.", {'target_action': 'pass', 'target_object': CUBE}],
    ]
    test_colors = [
    ["Podej mi červenou kostku", {'target_action': 'pass', 'target_object': CUBE, 'to_color': 'red'}],
    ["Podej mi zelenou kostku", {'target_action': 'pass', 'target_object': CUBE, 'to_color': 'green'}],
    ]
    test_object_storage = [
    ["Nalij kostku na kolo.", {'target_action': 'pour', 'target_object': CUBE, 'target_storage': WHEEL}],
    ["Nandej kostku na kolo.", {'target_action': 'stack', 'target_object': CUBE, 'target_storage': WHEEL}],
    ["Polož kostku na kolo.", {'target_action': 'put-into', 'target_object': CUBE, 'target_storage': WHEEL}],
    ]
    test_action = [
    ["Stop.", {'target_action': 'stop'}],
    ["Nahoru.", {'target_action': 'move-up'}],
    ["Pusť.", {'target_action': 'release'}],
    ]
    test_all_actions = [
    ["Postrč kostku.", {'target_action': 'push', 'target_object': CUBE}],
    ["Odlep kostku.", {'target_action': 'unglue', 'target_object': CUBE}],
    ]
    test_properties = [
    ["Nandej zelenou kostku na červený kostku.", {'target_action': 'stack', 'target_object': CUBE, 'to_color': 'green', 'target_storage': CUBE, 'ts_color': 'red'}],
    ]
    test_wrong_template = [
    ["Pusť kostku na šuplík", {'target_action': 'release', 'target_object': CUBE, 'target_storage': 'drawer'}],
    ["Červenou kostku", {'target_action': 'noop', 'target_object': CUBE, 'to_color': 'red'}],
    ["Červenou", {'target_action': 'noop', 'to_color': 'red'}],
    ["Kostku", {'target_action': 'noop', 'target_object': CUBE}],
    ["Podej mi", {'target_action': 'pass'}],
    ["Seber", {'target_action': 'pick'}],
    ]
    # example_list = [
    # ["Ukaž na zelenou kostku", ('point', 'green cube')],
    # ["Ukaž na kolík", ('', '')],
    # ["Ukaž na zelenou kostku", ('', '')],
    
    # ["Podej mi kladivo", ('Noop', '')],
    # ]
    '''example_list = [    
        # FUTURE?
        # ["Vezmi jablko a hrušku a dej je do krabice", (..)]
        ["Dej zelenou kostku na červenou kostku"]
    ]'''

class ReExSet(BaseSet):
    test_object_and_color = [
    ["Seber červenou kostku", {'target_action': 'pick', 'target_object': CUBE, 'to_color': 'red'}],
    ["Ukaž na červenou kostku", {'target_action': 'point', 'target_object': CUBE, 'to_color': 'red'}],
    ["Podej mi červenou kostku", {'target_action': 'pass', 'target_object': CUBE, 'to_color': 'red'}],
    ["Podej mi zelenou kostku.", {'target_action': 'pass', 'target_object': CUBE}],
    # ["Ukaž na modrý šuplík", {'target_action': 'point', 'target_object': 'drawer', 'to_color': 'blue'}],
    ]
    test_object = [
    ["Seber kostku", {'target_action': 'pick', 'target_object': CUBE}],
    ["Ukaž na kostku", {'target_action': 'point', 'target_object': CUBE}],
    ["Podej mi kostku", {'target_action': 'pass', 'target_object': CUBE}],
    ["Seber křupky", {'target_action': 'pick', 'target_object': 'crackers'}],
    ["Ukaž na křupky", {'target_action': 'point', 'target_object': 'crackers'}],
    ["Podej mi křupky", {'target_action': 'pass', 'target_object': 'crackers'}],
    ]
    # ["Seber destičku", {'target_action': 'pick', 'target_object': 'wafer'}],
    # ["Ukaž na destičku", {'target_action': 'point', 'target_object': 'wafer'}],
    # ["Podej mi destičku", {'target_action': 'pass', 'target_object': 'wafer'}],
    # ]
    # test_unknown_object = [
    # ["Seber kladivo", {'target_action': 'Noop'}],
    # ["Ukaž na kladivo", {'target_action': 'Noop'}],
    # ["Podej mi kladivo",{'target_action': 'Noop'}],
    # ]
    test_synonyms = [
    ["Zvedni kostku", {'target_action': 'pick', 'target_object': CUBE}],
    ["Získej kostku", {'target_action': 'pick', 'target_object': CUBE}],
    ["Dej mi kostku", {'target_action': 'pass', 'target_object': CUBE}],
    ["Dej mi krychli",{'target_action': 'pass', 'target_object': CUBE}],
    ["Dej mi krychli s dírama.", {'target_action': 'pass', 'target_object': CUBE}],
    ]
    test_shuffled = [
    ["Kostku mi dej.", {'target_action': 'pass', 'target_object': CUBE}],
    ["dej kostku", {'target_action': 'pass', 'target_object': CUBE}],    
    ]
    test_multi_sentence = [
    ["Kostku mi dej. A pak ukaž na kostku.", {'target_action': 'pass', 'target_object': CUBE}],
    ]
    test_colors = [
    ["Podej mi červenou kostku", {'target_action': 'pass', 'target_object': CUBE, 'to_color': 'red'}],
    ["Podej mi zelenou kostku", {'target_action': 'pass', 'target_object': CUBE, 'to_color': 'green'}],
    ]
    test_object_storage = [
    ["Nalij kostku na křupky.", {'target_action': 'pour', 'target_object': CUBE, 'target_storage': 'crackers'}],
    ["Nandej kostku na křupky.", {'target_action': 'stack', 'target_object': CUBE, 'target_storage': 'crackers'}],
    ["Polož kostku na křupky.", {'target_action': 'put-into', 'target_object': CUBE, 'target_storage': 'crackers'}],
    ]
    test_action = [
    ["Stop.", {'target_action': 'stop'}],
    ["Nahoru.", {'target_action': 'move-up'}],
    ["Pusť.", {'target_action': 'release'}],
    ]
    test_all_actions = [
    ["Postrč kostku.", {'target_action': 'push', 'target_object': CUBE}],
    ["Odlep kostku.", {'target_action': 'unglue', 'target_object': CUBE}],
    ]
    test_properties = [
    ["Nandej zelenou kostku na červený kostku.", {'target_action': 'stack', 'target_object': CUBE, 'to_color': 'green', 'target_storage': CUBE, 'ts_color': 'red'}],
    ]
    test_wrong_template = [
    # ["Pusť kostku na šuplík", {'target_action': 'release', 'target_object': CUBE, 'target_storage': 'drawer'}],
    ["Červenou kostku", {'target_action': 'noop', 'target_object': CUBE, 'to_color': 'red'}],
    ["Červenou", {'target_action': 'noop', 'to_color': 'red'}],
    ["Kostku", {'target_action': 'noop', 'target_object': CUBE}],
    ["Podej mi", {'target_action': 'pass'}],
    ["Seber", {'target_action': 'pick'}],
    ]
    # example_list = [
    # ["Ukaž na zelenou kostku", ('point', 'green cube')],
    # ["Ukaž na kolík", ('', '')],
    # ["Ukaž na zelenou kostku", ('', '')],
    
    # ["Podej mi kladivo", ('Noop', '')],
    # ]
    '''example_list = [    
        # FUTURE?
        # ["Vezmi jablko a hrušku a dej je do krabice", (..)]
        ["Dej zelenou kostku na červenou kostku"]
    ]'''    

class ReExSetEn(BaseSet):
    # test_object_and_color = [
    # ["pick red cube", {'target_action': 'pick', 'target_object': CUBE, 'to_color': 'red'}],
    # ["Point on red cube", {'target_action': 'point', 'target_object': CUBE, 'to_color': 'red'}],
    # ["Pass me a red cube", {'target_action': 'pass', 'target_object': CUBE, 'to_color': 'red'}],
    # ["Pass me a green cube.", {'target_action': 'pass', 'target_object': CUBE}],
    # # ["Ukaž na modrý šuplík", {'target_action': 'point', 'target_object': 'drawer', 'to_color': 'blue'}],
    # ]
    # test_object = [
    # ["Pick up cube", {'target_action': 'pick', 'target_object': CUBE}],
    # ["Point to a cube", {'target_action': 'point', 'target_object': CUBE}],
    # ["Pass me a cube", {'target_action': 'pass', 'target_object': CUBE}],
    # ["Pick up crackers", {'target_action': 'pick', 'target_object': 'crackers'}],
    # ["Point on a crackers", {'target_action': 'point', 'target_object': 'crackers'}],
    # ["Pass me a crackers", {'target_action': 'pass', 'target_object': 'crackers'}],
    # ]
    # # ["Seber destičku", {'target_action': 'pick', 'target_object': 'wafer'}],
    # # ["Ukaž na destičku", {'target_action': 'point', 'target_object': 'wafer'}],
    # # ["Podej mi destičku", {'target_action': 'pass', 'target_object': 'wafer'}],
    # # ]
    # # test_unknown_object = [
    # # ["Seber kladivo", {'target_action': 'Noop'}],
    # # ["Ukaž na kladivo", {'target_action': 'Noop'}],
    # # ["Podej mi kladivo",{'target_action': 'Noop'}],
    # # ]
    # test_synonyms = [
    # ["Pick cube", {'target_action': 'pick', 'target_object': CUBE}],
    # ["Pick up a cube", {'target_action': 'pick', 'target_object': CUBE}],
    # # ["Dej mi kostku", {'target_action': 'pass', 'target_object': CUBE}],
    # # ["Dej mi krychli",{'target_action': 'pass', 'target_object': CUBE}],
    # # ["Dej mi krychli s dírama.", {'target_action': 'pass', 'target_object': CUBE}],
    # ]
    # test_shuffled = [
    # ["Cube pass me it", {'target_action': 'pass', 'target_object': CUBE}],
    # ["pass cube", {'target_action': 'pass', 'target_object': CUBE}],    
    # ]
    # test_multi_sentence = [
    # ["Cube pass me. And then point on a cube.", {'target_action': 'pass', 'target_object': CUBE}],
    # ]
    # test_colors = [
    # ["Pass me a red cube", {'target_action': 'pass', 'target_object': CUBE, 'to_color': 'red'}],
    # ["Pass me a green cube", {'target_action': 'pass', 'target_object': CUBE, 'to_color': 'green'}],
    # ]
    # test_object_storage = [
    # ["Pour cube on a crackers.", {'target_action': 'pour', 'target_object': CUBE, 'target_storage': 'crackers'}],
    # ["Stack cube on a crackers.", {'target_action': 'stack', 'target_object': CUBE, 'target_storage': 'crackers'}],
    # ["Put cube on crackers.", {'target_action': 'put-into', 'target_object': CUBE, 'target_storage': 'crackers'}],
    # ]
    # test_action = [
    # ["Stop.", {'target_action': 'stop'}],
    # ["Move up.", {'target_action': 'move-up'}],
    # ["Release.", {'target_action': 'release'}],
    # ]
    # test_all_actions = [
    # ["Push cube.", {'target_action': 'push', 'target_object': CUBE}],
    # ["Unglue cube.", {'target_action': 'unglue', 'target_object': CUBE}],
    # ]
    # test_properties = [
    # ["Stack green cube on red cube.", {'target_action': 'stack', 'target_object': CUBE, 'to_color': 'green', 'target_storage': CUBE, 'ts_color': 'red'}],
    # ]
    # test_wrong_template = [
    # # ["Pusť kostku na šuplík", {'target_action': 'release', 'target_object': CUBE, 'target_storage': 'drawer'}],
    # ["Red cube", {'target_action': 'noop', 'target_object': CUBE, 'to_color': 'red'}],
    # ["Red", {'target_action': 'noop', 'to_color': 'red'}],
    # ["Cube", {'target_action': 'noop', 'target_object': CUBE}],
    # ["Pass me", {'target_action': 'pass'}],
    # ["Pick", {'target_action': 'pick'}],
    # ]

    test_dataset = [
    ["stack cleaner to crackers", {'target_action': 'stack', 'target_object': "cleaner", 'target_storage': "crackers"}],
    ["stack cleaner to cube", {'target_action': 'stack', 'target_object': "cleaner", 'target_storage': "cube"}],
    ["stack cleaner to cup", {'target_action': 'stack', 'target_object': "cleaner", 'target_storage': "cup"}],
    ["stack can to cleaner", {'target_action': 'stack', 'target_object': "can", 'target_storage': "cleaner"}],
    ["unglue a can can can", {'target_action': 'unglue', 'target_object': "can"}],
    ["push cube", {'target_action': 'push', 'target_object': "cube"}],
    ["pick cleaner", {'target_action': 'pick', 'target_object': "cleaner"}],
    ["unglue box", {'target_action': 'unglue', 'target_object': "box"}],
    ["push can", {"target_action": "push", "target_object": "can"}],
    ["put_into cleaner to drawer", {"target_action": "put-into", "target_object": "cleaner", "target_storage": "drawer_socket"}],
    ["unglue cleaner", {"target_action": "unglue", "target_object": "cleaner"}],
    ["pour can to bowl", {"target_action": "pour", "target_object": "can", "target_storage": "bowl"}],
    ["stack cleaner to crackers", {"target_action": "stack", "target_object": "cleaner", "target_storage": "crackers"}],
    ["put_into cube to drawer", {"target_action": "put-into", "target_object": "cube", "target_storage":"drawer_socket"}],
    ]
    # example_list = [
    # ["Ukaž na zelenou kostku", ('point', 'green cube')],
    # ["Ukaž na kolík", ('', '')],
    # ["Ukaž na zelenou kostku", ('', '')],
    
    # ["Podej mi kladivo", ('Noop', '')],
    # ]
    '''example_list = [    
        # FUTURE?
        # ["Vezmi jablko a hrušku a dej je do krabice", (..)]
        ["Dej zelenou kostku na červenou kostku"]
    ]'''    

    
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
    # Success (cube_holes, wheel, wafer visible)
        
    # ts = TestSet()
    # ts = ReExSet()
    ts = ReExSetEn()
    test_examples_sorted = ts()
    print(f"Test set printout:\n{ts}")

    for set_name in test_examples_sorted.keys():
        print(f"New set: {set_name}")
        for sentence, solution in test_examples_sorted[set_name]:
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
                ret = URIRef(uri).fragment.split("_od_")[0]
                if ret == "":
                    return uri
                else:
                    return ret

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
    

# test_target_objects_detector()
if __name__ == '__main__':
    test_nlp_1()
