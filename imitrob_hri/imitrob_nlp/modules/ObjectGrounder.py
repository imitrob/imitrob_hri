#!/usr/bin/env python
"""
Copyright (c) 2019 CIIRC, CTU in Prague
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.

@author: Zdenek Kasner, Karla Štěpánová
@mail:  karla.stepanova@cvut.cz
"""
from enum import Enum
from typing import ClassVar, Any

# from crow_nlp.nlp_crow.database.Database import Database
# from crow_nlp.nlp_crow.database.DatabaseAPI import DatabaseAPI
from imitrob_hri.imitrob_nlp.modules.UserInputManager import UserInputManager
from imitrob_hri.imitrob_nlp.modules.ColorDetector import ColorDetector
# from crow_ontology.crowracle_client import CrowtologyClient
# from rdflib.namespace import Namespace, RDF, RDFS, OWL, FOAF, XSD


# ONTO_IRI = "http://imitrob.ciirc.cvut.cz/ontologies/crow"
# CROW = Namespace(f"{ONTO_IRI}#")

import logging
import owlready2 as ow

# db = Database()

# with db.onto as onto:
class ObjectGrounder:
    # namespace = db.onto
    #class Flags(Enum):
    #     CAN_BE_PICKED = 1

    def __init__(self, language = 'en', client = None):
        self.lang = language
        # self.db_api = DatabaseAPI()
        self.crowracle = client
        self.cd = ColorDetector(client = self.crowracle)
        self.ar = UserInputManager(language = self.lang)
        #self.onto = self.crowracle.onto

        self.logger = logging.getLogger(__name__)
        self.templ_file = self.ar.load_file('templates_detection.json')
        self.obj_det_file = self.ar.load_file('objects_detection.json')
        self.guidance_file = self.ar.load_file('guidance_dialogue.json')

    def ground_object(self, obj_placeholder, flags=()) -> Any:
        """Returns real grounded object, real object from the scene

        Args:
            obj_placeholder (ObjectDetectedData): Result of Match/ObjectDetector, 
            it returns semantic information about possible objects
            flags (tuple, optional): _description_. Defaults to ().

        Returns:
            ObjectsGroundedData(): _description_
        """        

        # try to find out the class of the object
        # DEL: the 0th item of an is_a list should be always ObjectPlaceholder
        # DEL: if the item is bound to a real class, the class will be the 1st item
        if obj_placeholder is None:
            # self.ar.buffered_say(self.guidance_file[self.lang]["nonsense_ground_object_none"])
            return
        else:
            pass

        # if "last_mentioned" in obj_placeholder.flags:
        #     #objs = [self.get_last_mentioned_object()]
        #     print('add get_last_mentioned_object function')
        #     return
        # else:
        objs = self.crowracle.getTangibleObjects()
        obj = None
        objL = []
        objLC = []
        objLC_probs = []

        for objI in objs:
            # TODO now checking only the first class from the mentioned ones, but later we want to check all the classes that were mentioned
            if self.crowracle.get_nlp_from_uri(objI)[0]== obj_placeholder.objs_mentioned_cls:
                objL.append(objI)
                print('this is color of the object')
                print(obj_placeholder.objs_properties['color'])
                if len(obj_placeholder.objs_properties['color'].keys()) != 0:
                    if list(obj_placeholder.objs_properties['color'].keys())[0] == self.crowracle.get_nlp_from_uri(self.crowracle.get_color_of_obj(objI))[0]:
                        objLC.append(objI)
                        objLC_probs.append(1)
                    else: # TODO if not matching color, we still save the target object as detected, but add it smaller prob - can be changed
                        objLC.append(objI)
                        objLC_probs.append(0.5)
                else:
                    print('no color of object specified')
                    objLC.append(objI)
                    objLC_probs.append(1)
        #if len(objLC) != 0:
        #    obj = objLC # return all found, not the first one - ask for specification or choose randomly in logic node
            # TODO user feedback should be moved after modality merger
#             if len(objLC) > 1:
#                 self.ar.buffered_say(self.guidance_file[self.lang]["object_found_multi_ws"] + self.obj_det_file[self.lang][self.crowracle.get_nlp_from_uri(obj[0])[0]], say = 2)
# #                self.ar.say(self.guidance_file[self.lang]["object_found_multi_ws"] + self.crowracle.get_nlp_from_uri(obj)[0])
#             else:
#                 #self.ar.say('found an object in the workspace')
#                 self.ar.buffered_say(
#                     self.guidance_file[self.lang]["object_found_ws"] + self.obj_det_file[self.lang][self.crowracle.get_nlp_from_uri(obj[0])[0]])
        # elif len(objLC) == 0 and len(objL) != 0:
                #TODO not only the first one, but list of available ones
                # self.ar.buffered_say(self.guidance_file[self.lang]["no_object_color_found"] + self.templ_file[self.lang][
                #    obj_placeholder.color[0]] + " " +
                            # self.obj_det_file[self.lang][self.crowracle.get_nlp_from_uri(obj_placeholder.is_a[0])[0]] + " " +
                            # self.guidance_file[self.lang]["dif_color_found"] +
                            # self.templ_file[self.lang][
                            #     self.crowracle.get_nlp_from_uri(self.crowracle.get_color_of_obj(objL[0]))[0]] + " " +
                            # self.obj_det_file[self.lang][self.crowracle.get_nlp_from_uri(objL[0])[0]], say = 2
                            # )
        # elif obj == None:
        #     self.logger.warning(self.guidance_file[self.lang]["no_object_workspace"] + self.obj_det_file[self.lang][self.crowracle.get_nlp_from_uri(obj_placeholder.is_a[0])[0]])
        #     self.ar.buffered_say(self.guidance_file[self.lang]["no_object_workspace"] + self.obj_det_file[self.lang][self.crowracle.get_nlp_from_uri(obj_placeholder.is_a[0])[0]], say = 2)  # f"{cls_obj}")
        #     self.ar.buffered_say(self.guidance_file[self.lang]["no_object_workspace_place"], say = 3)

        # if obj:
            # self.logger.debug(f"Object found for {obj_placeholder}: {obj[0]}")
            #self.ar.say(self.guidance_file[self.lang]["object_found"],f"{obj_placeholder}")
#        is_a, col, loc = self.get_obj_placeholder_attr(obj_placeholder)
        ''' TODO: 
        objects_grounded_data = ObjectsGroundedData(objects_detected_data=obj_placeholder)
        objects_grounded_data.objLC = objLC
        objects_grounded_data.objLC_probs = objLC_probs
        '''
        return objLC, objLC_probs, obj_placeholder.objs_mentioned_cls, obj_placeholder.objs_mentioned_cls_probs, obj_placeholder.objs_properties
        return objects_grounded_data

    # def get_obj_placeholder_attr(self, obj_ph):
    #     if len(obj_ph.is_a) > 0:
    #         is_a = obj_ph.is_a[-1]
    #     else:
    #         is_a = None
    #     if len(obj_ph.color) > 0:
    #         col = obj_ph.color[-1]
    #     else:
    #         col = None
    #     if len(obj_ph.location) > 0:
    #         loc = obj_ph.location[-1]
    #     else:
    #         loc = None
    #     return is_a, col, loc

    def get_prop_name(self, prop : ow.DataPropertyClass):
        # we need to use the python name of properties whenever it is defined
        if hasattr(prop, "python_name"):
            return prop.python_name

        return prop.name