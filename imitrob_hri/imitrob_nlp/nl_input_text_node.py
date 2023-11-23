#!/usr/bin/env python3
"""
Copyright (c) 2019 CIIRC CTU in Prague
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.

@author: Karla Stepanova
@mail: karla.stepanova@cvut.cz
"""
import rclpy
from rclpy.node import Node
from ros2param.api import call_get_parameters
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from crow_msgs.msg import SentenceProgram
import message_filters

import curses
import time
import numpy as np
from datetime import datetime
from curses.textpad import Textbox, rectangle
from crow_ontology.crowracle_client import CrowtologyClient
from rdflib.namespace import Namespace, RDF, RDFS, OWL, FOAF, XSD
from rdflib import URIRef, BNode, Literal, Graph
from rdflib.term import Identifier
from crow_params.client import ParamClient

ONTO_IRI = "http://imitrob.ciirc.cvut.cz/ontologies/crow"
CROW = Namespace(f"{ONTO_IRI}#")

class NLInput(Node):
    # listens to /averaged_markers from object_detection package and in parallel to button presses. When there is an approval
    # to collect cubes (key a -> y), robot picks up the currently detected cubes and places them to storage
    def __init__(self, node_name="nl_input"):
        super().__init__(node_name)
        self.crowracle = CrowtologyClient(node=self)
        self.onto = self.crowracle.onto
        self.pclient = ParamClient()

        # add the red cube to the ontology
        #red_cube = self.onto.makeEntity(CROW.meinhasenfuss, {CROW.hasColor: CROW.COLOR_RED, CROW.x: Literal(13), CROW.y: Literal(7)})
        #print(red_cube.CROW.hascolor)

        # self.add_object_individual(0, [1.0, 2.0, 3.0], 'cube_holes_1','Cube')
        # self.add_object_individual(0, [1.0, 1.0, 3.0], 'cube_holes_2','Cube')
        # self.add_object_individual(0, [1.0, 4.0, 3.0], 'cube_holes_3','Cube')
        # self.add_object_individual(0, [1.0, 4.0, 2.0], 'peg_1','Peg')
        # self.add_object_individual(0, [1.0, 4.0, 2.0], 'hammer')


        input_sentences = [
            #     "Ukaž na červenou kostku",
            # "Ukaž na modrý kolík",
            # "Ukaž na zelenou kostku",
            # "Ukaž na kolík",
            # "Ukaž na zelenou kostku"
            #"Definuj modrou pozici"
            #"Pick a red cube",
            'Seber kostku',
            ]

        # input_sentences = ["Point to the red cube",
        #                "Point to the blue peg",
        #                "Point to the green cube",
        #                "Pick up a red cube"]

    # input_sentences = [
        #     ##"Give me a cube",
        #     #"Put a cube on position 0 0",
        #     "Point to the red cube",
        #     # "Pick a red cube",
        #     # #"Take any cube",
        #     # "Glue a point here",
        #     # "Glue a point on position 4 3",
        #     # "Glue a point in the center of a panel",
        #     # "Glue a point here and glue a point on position 4 3",
        #     # #"Glue a point in the center of the blue panel",
        #      "Point to the blue peg",
        #     "Point to the green cube"
        #     # "Pick a red cube and pick a green cube"
        #     # #"Put a red cube on position 0 0",
        #     # #"Take any cube",
        #     #  #"Put it down",
        #     # #"Learn a new task called glue a point on the panel",
        #     # #"Take a glue. Glue a point in the center of the panel. Put the glue on position three five.",
        #     # #"Glue a point on the panel",
        #     # #"Glue a point in the center of the panel and glue a point on the right side of the panel"
        #     # #'Pick a red cube'
        #     # #"First make a point here and then make a point here"
        # ]

        # input_sentences = [
        #     "Define a new area called storage 1 with corners 0 0, 0 2, 2 2, 2 0",
        #     "Glue a point in the center of storage 1",
        #     "Learn to build a tower called tower 1",
        #     "Build tower 1",
        #     #"Show actions learned from demonstration",
        #     "Glue a point here",
        #     #"Glue a point in the center of the panel",
        #     #"Glue a point on the right side of the panel",
        #     #"Pick a red cube",
        #     #"Put it on top of the green cube.",
        #     #"Take any cube",
        #     #"Put it down",
        #     #"Learn a new task called glue a point on the panel",
        #     #"Take a glue. Use it to glue a point in the center of the panel. Put the glue on position three five.",
        #     #"Glue a point on the panel and take any cube."
        # ]
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        self.nl_publisher = self.create_publisher(SentenceProgram, "/nl_input", qos)
        msg = SentenceProgram()
        self.pclient.nlp_ongoing = True
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.data = input_sentences
        print('will publish {}',msg.data)
        self.nl_publisher.publish(msg)
        print('published {}',msg.data)

    def add_object_individual(self, id, loc, name, typeN):
        self.id = id
        location = loc
        individual_name = name

        PART = Namespace(f"{ONTO_IRI}/{individual_name}#")  # Add object to database
        if typeN == 'Cube':
            self.onto.add((CROW[individual_name], RDF.type, CROW.Cube))  # Add ID
        if typeN == 'Peg':
            self.onto.add((CROW[individual_name], RDF.type, CROW.Peg))  # Add ID

        self.onto.add(
            (CROW[individual_name], CROW.hasId, Literal('od_' + str(self.id), datatype=XSD.string)))  # Add color
        self.onto.set((CROW[individual_name], CROW.hasColor, CROW.COLOR_RED))  # Add AbsoluteLocaton
        prop_name = PART.xyzAbsoluteLocation
        prop_range = list(self.onto.objects(CROW.hasAbsoluteLocation, RDFS.range))[0]
        self.onto.add((prop_name, RDF.type, prop_range))
        self.onto.add((prop_name, CROW.x, Literal(location[0], datatype=XSD.float)))
        self.onto.add((prop_name, CROW.y, Literal(location[1], datatype=XSD.float)))
        self.onto.add((prop_name, CROW.z, Literal(location[2], datatype=XSD.float)))
        self.onto.set((CROW[individual_name], CROW.hasAbsoluteLocation, prop_name))

def main():
    rclpy.init()
    time.sleep(1)

    sentence = NLInput()

    rclpy.spin(sentence)
    sentence.destroy_node()

    # spin() simply keeps python from exiting until this node is stopped

if __name__ == '__main__':
    main()
