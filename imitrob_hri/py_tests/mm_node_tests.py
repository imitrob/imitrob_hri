
import rclpy, time, json
from rclpy.node import Node
from teleop_msgs.msg import HRICommand

from imitrob_templates.small_ontology_scene_reader import SceneOntologyClient

MM_G_IN = "/gg/hri_command"
MM_L_IN = "/nlp/hri_command"
MM_OUT = "/mm/solution"

class PublishHRICommand(Node):
    def __init__(self):
        super().__init__("rosnodePublisherOfHRICommand")
        
        self.pub_g = self.create_publisher(HRICommand, MM_G_IN, 5)
        self.pub_l = self.create_publisher(HRICommand, MM_L_IN, 5)

        self.soc = SceneOntologyClient(self)

        self.sub = self.create_subscription(HRICommand, MM_OUT, self.mm_solution_clb, 5)
        self.mm_solution_queue = []

    def mm_solution_clb(self, msg):
        self.mm_solution_queue.append(msg)

def main():
    rclpy.init()
    rosnode = PublishHRICommand()
    
    print("This is imitrob_template tester")
    print("- This exmaple uses crow object cube_holes!")

    while True:
        target_action = str(input("Enter task name: "))

        if target_action not in ['release']:
            print("Need to ground scene object:")

            scene = rosnode.soc.get_scene2()
            target_object = None
            for o_name in scene.O:
                if 'cube_holes' in o_name or 'cube' in o_name: # e.g.: 'cube_holes_od_0'
                    target_object = o_name
                    break
            if target_object is None:
                print("cube (cube_holes) not found on the scene! Waiting and trying again")
                time.sleep(5)
                continue
            else:
                print(f"cube chosen ({target_object})")

        else:
            print("Don't need to ground scene object")
            target_object = ''

        msg_to_send = HRICommand()
        s = "{'target_action': '" + target_action + "', 'target_object': '" + target_object + "', "
        s += "'actions': ['" + target_action + "', 'release', 'pass', 'point'], " #"', 'move_left', 'move_down', 'move_right', 'pick_up', 'put', 'place', 'pour', 'push', 'replace'], "
        s += "'action_probs': ['" + "1.0" + "', '0.05', '0.1', '0.15'], " #", 0.014667431372768347, 0.0008680663118536268, 0.035168211530459945, 0.0984559292675215, 0.012854139530004692, 0.0068131722011598225, 0.04846120672655781, 0.0020918881693065285, 0.01454853390045828], "
        
        if target_object == '':
            scene_objects = "[]"
            scene_object_probs = "[]"
            scene_object_classes = "[]"
        else:
            scene_objects = "['" + target_object + "', 'wheel', 'sphere']"
            scene_object_probs = "[1.0, 0.1, 0.15]"
            scene_object_classes = "['object']"
        
        s += "'action_timestamp': 0.0, 'objects': " + scene_objects + ", "
        s += "'object_probs': " + scene_object_probs + ", 'object_classes': " + scene_object_classes + ", 'parameters': ''}"
        s = s.replace("'", '"')
        msg_to_send.data = [s]
        
        rosnode.pub_g.publish(msg_to_send)
        time.sleep(0.5)
        rosnode.pub_l.publish(msg_to_send)
        
        while len(rosnode.mm_solution_queue) == 0:
            rclpy.spin_once(rosnode)
            time.sleep(1)
            print("still no feedback")
        msg = rosnode.mm_solution_queue.pop(0)
        print(msg)
        msg_str_to_parse = msg.data[0]
        msg_parsed = json.loads(msg_str_to_parse)
        print(msg_parsed)

if __name__ == '__main__':
    main()