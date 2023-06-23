

input('''1.
Create Ontology Client
''')
from crow_ontology.crowracle_client import CrowtologyClient
crowracle = CrowtologyClient()

crowracle.onto

input('''

''')
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from crow_msgs.msg import SentenceProgram
from rclpy.node import Node

rosnode = Node("node")

qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
rosnode.nl_publisher = rosnode.create_publisher(SentenceProgram, "/nl_input", qos)
msg = SentenceProgram()
msg.header.stamp = rosnode.get_clock().now().to_msg()
msg.data = [
            # "Ukaž na červenou kostku",
            # "Ukaž na modrý kolík",
            # "Ukaž na zelenou kostku",
            # "Ukaž na kolík",
            # "Ukaž na zelenou kostku"
            "Definuj modrou pozici"
]

from rdflib.namespace import Namespace, RDF, RDFS, OWL, FOAF, XSD
from rdflib import URIRef, BNode, Literal, Graph
ONTO_IRI = "http://imitrob.ciirc.cvut.cz/ontologies/crow"
CROW = Namespace(f"{ONTO_IRI}#")

def add_object(self):
    o = self.onto
    id = 0
    location = [0.,0.,0.]
    individual_name = 0

    PART = Namespace(f"{ONTO_IRI}/{individual_name}#")  # Add object to database
    #if typeN == 'Cube':
    o.add((CROW[individual_name], RDF.type, CROW.Cube))  # Add ID
    #if typeN == 'Peg':
    #    o.add((CROW[individual_name], RDF.type, CROW.Peg))  # Add ID

    o.add( (CROW[individual_name], CROW.hasId, Literal('od_' + str(id), datatype=XSD.string)))  # Add color
    o.set((CROW[individual_name], CROW.hasColor, CROW.COLOR_RED))  # Add AbsoluteLocaton
    prop_name = PART.xyzAbsoluteLocation
    prop_range = list(o.objects(CROW.hasAbsoluteLocation, RDFS.range))[0]
    o.add((prop_name, RDF.type, prop_range))
    o.add((prop_name, CROW.x, Literal(location[0], datatype=XSD.float)))
    o.add((prop_name, CROW.y, Literal(location[1], datatype=XSD.float)))
    o.add((prop_name, CROW.z, Literal(location[2], datatype=XSD.float)))
    o.set((CROW[individual_name], CROW.hasAbsoluteLocation, prop_name))

add_object(crowracle.onto)

input('''
Understanding the OntologyAPI



- test 4
''')

import rclpy
from rclpy.node import Node
import curses
from curses.textpad import Textbox, rectangle
from crow_ontology.crowracle_client import CrowtologyClient
from rdflib.namespace import Namespace, RDF, RDFS, OWL, FOAF
from rdflib import URIRef, BNode, Literal, Graph
from rdflib.term import Identifier
import time

class OntoTester(Node):
    def __init__(self, node_name="onto_tester"):
        super().__init__(node_name)
        self.crowracle = CrowtologyClient(node=self)
        self.onto = self.crowracle.onto
        # self.get_logger().info(self.onto)

rclpy.init()
ot = OntoTester()
self = ot

self.get_logger().info("Found these classes of tangible objects in the database")
start = time.time()
qres = self.crowracle.getTangibleObjectClasses()
for c in qres:
    self.get_logger().info(f"{c}")
print(time.time() - start)
input()

self.get_logger().info("Found these objects in the scene (onto.triples)")
start = time.time()
res = list(self.onto.triples((None, self.crowracle.CROW.hasId, None)))
self.get_logger().info(f"{res}")
print(time.time() - start)
input("Note: Didn't found any object, even when I added object above")

self.get_logger().info("Found these tangible objects in the scene (getTangibleObjects)")
start = time.time()
self.get_logger().info(str(self.crowracle.getTangibleObjects()))
print(time.time() - start)
input("Found .../crow#")

print('get_uri_from_nlp')
uris = self.crowracle.get_uri_from_nlp('blue')
print(str(uris))

uris = self.crowracle.get_uri_from_nlp('cube')
print(str(uris))

print('get_nlp_from_uri')
names = self.crowracle.get_nlp_from_uri(self.crowracle.CROW.COLOR_BLUE)
print(names)
names = self.crowracle.get_nlp_from_uri(self.crowracle.CROW.CUBE)
print(names)
names = self.crowracle.get_nlp_from_uri(self.crowracle.CROW.Peg)
print(names)
names = self.crowracle.get_nlp_from_uri(self.crowracle.CROW.Human)
print(names)

print('get_all_tangible_nlp')
names = self.crowracle.get_all_tangible_nlp(language='EN')
print(names)

print('get_color_of_obj')
uri = self.crowracle.get_color_of_obj(self.crowracle.CROW.CUBE)
print(uri)

print('get_color_of_obj_nlp')
names = self.crowracle.get_color_of_obj_nlp('cube')
print(names)

print('get_obj_of_color')
uris = self.crowracle.get_obj_of_color(self.crowracle.CROW.COLOR_GREEN)
print(uris)

print('get_obj_of_color_nlp')
names = self.crowracle.get_obj_of_color_nlp('gold', all=False)
print(names)

print('get_location_of_obj')
uris = self.crowracle.get_location_of_obj(self.crowracle.CROW.CUBE)
print(uris)

print('find_obj_nlp')
locations = self.crowracle.find_obj_nlp('sphere', all=False)
print(locations)

print('find_obj_of_color_nlp')
locations = self.crowracle.find_obj_of_color_nlp('gold', all=False)
print(locations)

print('get_colors')
colors = self.crowracle.getColors()
print(colors)

print('get_colors_nlp')
colors = self.crowracle.get_colors_nlp()
print(colors)

print('get_pcl_dimensions_of_obj')
dims = self.crowracle.get_pcl_dimensions_of_obj(self.crowracle.CROW.cube_holes_od_2)
print(dims)

print('get_fixed_dimensions_of_obj')
dims = self.crowracle.get_fixed_dimensions_of_obj(self.crowracle.CROW.cube_holes_od_2)
print(dims)

q1 = {"color": self.crowracle.CROW.COLOR_LIGHT_BLUE}
q2 = {"color": None}
obj_cls1 = self.crowracle.CROW.Nut
obj_cls2 = None
print('get_obj_of_properties')
uris = self.crowracle.get_obj_of_properties(obj_cls1, q1, all=True)
print(uris)
print('get_obj_of_properties')
uris = self.crowracle.get_obj_of_properties(obj_cls2, q2, all=True)
print(uris)

print('get_obj_of_id')
objs = self.crowracle.get_obj_of_id('od_0')
print(objs)

print('get_name_from_prop')
name = self.crowracle.get_name_from_prop(self.crowracle.CROW.hasColor)
print(name)

print('get_prop_from_name')
uri = self.crowracle.get_prop_from_name('color')
print(uri)

print('get_all_tools')
uri = self.crowracle.get_all_tools(all=True)
print(uri)

print('get_all_tools')
uri = self.crowracle.get_all_tools(all=False)
print(uri)

print('get_tang_with_props')
list_of_dicts = self.crowracle.getTangibleObjectsProps()
print(list_of_dicts)

print('get_filter_object_properties')
dict_of_dicts = self.crowracle.get_filter_object_properties()
print(dict_of_dicts)

print('getMarkerGroupProps')
dicti = self.crowracle.getMarkerGroupProps('blue')
print(dicti)

print('get all actions')
dicti = self.crowracle.getActionsProps()
print(dicti)

print('get current actions')
dicti = self.crowracle.getCurrentAction()
print(dicti)

print('get obj of properties holding')
q1 = {"name": 'Holding something'}
uris = self.crowracle.get_obj_of_properties(self.crowracle.CROW.Action, q1, all=True)
print(uris)

print('get_position')
q1 = {"name": 'test pozice'}
uris = self.crowracle.get_obj_of_properties(self.crowracle.CROW.Position, q1, all=True)
print(uris)

self.crowracle.add_storage_space('sklad úložiště', [[0,0,0.2],[1,0,0.2],[1,1,0.2],[0,1,0.2]], [[0,0,0.2],[1,0,0.2],[1,1,0.2],[0,1,0.2],[0,0,0.4],[1,0,0.4],[1,1,0.4],[0,1,0.4]], 1, 1, [0.5,0.5,0.2])
self.crowracle.add_storage_space('box úložiště', [[0,0,0.2],[1,0,0.2],[1,1,0.2],[0,1,0.2]], [[0,0,0.2],[1,0,0.2],[1,1,0.2],[0,1,0.2],[0,0,0.4],[1,0,0.4],[1,1,0.4],[0,1,0.4]], 1, 1, [0.2,0.2,0.2])
self.crowracle.add_position('auto pozice', [0.5,0.5,0.5])
self.crowracle.add_detected_object('my_cube0', [0.1, 0.9, 0.3], [1,1,1], 'uuid0', '2021-07-16Z23:00:00', self.crowracle.CROW.CUBE, 'ukl')
self.crowracle.add_detected_object('my_cube1', [0.1, 0.8, 0.3], [1,1,1], 'uuid1', '2021-07-16Z23:00:00', self.crowracle.CROW.CUBE, 'ukl')
self.crowracle.add_detected_object('my_cube2', [0.1, 0.7, 0.3], [1,1,1], 'uuid2', '2021-07-16Z23:00:00', self.crowracle.CROW.CUBE, 'ukl')
self.crowracle.add_detected_object('my_cube3', [-0.1, 0.9, 0.3], [1,1,1], 'uuid3', '2021-07-16Z23:00:00', self.crowracle.CROW.CUBE, 'ukl')

print('get all storages')
q1 = {}
area_uri = self.crowracle.get_obj_of_properties(self.crowracle.CROW.StorageSpace, q1, all=True)
print(area_uri)
dicts = self.crowracle.getStoragesProps()
print(dicts)

print('positions props')
dicts = self.crowracle.getPositionsProps()
print(dicts)

print('get_storage blue')
q1 = {"name": 'modrá úložiště'}
area_uri = self.crowracle.get_obj_of_properties(self.crowracle.CROW.StorageSpace, q1, all=True)
print(area_uri)

print('test_obj_in_area')
obj_uri = self.crowracle.getTangibleObjects()
print(obj_uri)
res = self.crowracle.test_obj_in_area(obj_uri[0], area_uri)
print(res)
res = self.crowracle.test_obj_in_area(obj_uri[-1], area_uri)
print(res)

print('get objs in area')
uris = self.crowracle.get_objs_in_area(area_uri)
print(uris)

print('get free space area')
space = self.crowracle.get_free_space_area(area_uri)
print(space)

print('get centroid of area')
centroid = self.crowracle.get_area_centroid(area_uri)
print(centroid)

print('done')




