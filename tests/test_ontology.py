

from imitrob_templates.small_ontology_scene_reader import SceneOntologyClient
import rclpy
from rclpy.node import Node

def test_scene_ontology_client():
    
    rclpy.init()
    rosnode = Node('ontology_reader_node')
    soc = SceneOntologyClient(rosnode)
    s2 = soc.get_scene2()
    print("Scene 2")
    print(f"{s2}")
    s3 = soc.get_scene3()
    print("Scene 3")
    print(f"{s3}")
    print("Done")
    
    print(s2.object_types)
    
    
    rclpy.shutdown()
    
def test_if_it_sees_the_cube():
    rclpy.init()
    rosnode = Node('ontology_reader_node')
    soc = SceneOntologyClient(rosnode)
    s2 = soc.get_scene2()

    assert s2.get_object_by_type("cube_holes") is not None 
    
    rclpy.shutdown()    


if __name__ == '__main__':
    test_scene_ontology_client()