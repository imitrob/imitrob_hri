#!/usr/bin/env python3
"""
Copyright (c) 2019 CIIRC, CTU in Prague
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.

@author: Zdenek Kasner
"""
#from crow_nlp.nlp_crow.database.Database import Database
# from crow_nlp.nlp_crow.templates.actions.DefineAreaAction import DefineAreaAction
# from crow_nlp.nlp_crow.templates.actions.DemonstrationListAction import DemonstrationListAction
# from crow_nlp.nlp_crow.templates.actions.LearnNewTaskAction import LearnNewTaskAction
# from crow_nlp.nlp_crow.templates.actions.LearnTowerFromDemonstrationAction import LearnTowerFromDemonstrationAction
# from crow_nlp.nlp_crow.templates.tasks.ApplyGlueTask import ApplyGlueTask
from imitrob_templates.templates.PickTask import PickTask
# from crow_nlp.nlp_crow.templates.tasks.TidyTask import TidyTask
from imitrob_templates.templates.PointTask import PointTask
from imitrob_templates.templates.PassTask import PassTask
# from crow_nlp.nlp_crow.templates.tasks.FetchTask import FetchTask
# from crow_nlp.nlp_crow.templates.tasks.FetchToTask import FetchToTask
# from crow_nlp.nlp_crow.templates.tasks.ReleaseTask import ReleaseTask
# from crow_nlp.nlp_crow.templates.tasks.StopTask import StopTask
# from crow_nlp.nlp_crow.templates.actions.DefineStorageAction import DefineStorageAction
# from crow_nlp.nlp_crow.templates.actions.DefinePositionAction import DefinePositionAction
# from crow_nlp.nlp_crow.templates.actions.RemoveCommandLast import RemoveCommandLast
# from crow_nlp.nlp_crow.templates.actions.RemoveCommandX import RemoveCommandX
# from crow_nlp.nlp_crow.templates.actions.BuildAssembly import BuildAssembly
# from crow_nlp.nlp_crow.templates.actions.CancelAssembly import CancelAssembly
# from crow_nlp.nlp_crow.templates.actions.RemoveProduct import RemoveProduct

from enum import Enum

# from crow_nlp.nlp_crow.templates.tasks.PutTask2 import PutTask


class TemplateType(Enum):
    # mapping from constants to classes
    PICK_TASK = PickTask
    # APPLY_GLUE = ApplyGlueTask
    # PUT_TASK = PutTask
    # TIDY_TASK = TidyTask
    POINT_TASK = PointTask
    PASSME_TASK = PassTask
    # DEFINE_STORAGE = DefineStorageAction
    # DEFINE_POSITION = DefinePositionAction
    # REMOVE_COMMAND_LAST = RemoveCommandLast
    # REMOVE_COMMAND_X = RemoveCommandX
    # FETCH = FetchTask
    # FETCH_TO = FetchToTask
    # RELEASE = ReleaseTask
    # STOP = StopTask
    # BUILD = BuildAssembly
    # BUILD_CANCEL = CancelAssembly
    # PRODUCT_REMOVE = RemoveProduct
    # LEARN_NEW_TASK = LearnNewTaskAction
    # LEARN_TOWER = LearnTowerFromDemonstrationAction
    # DEMONSTRATION_LIST = DemonstrationListAction
    # DEFINE_AREA = DefineAreaAction


#db = Database()


# with db.onto as onto:
class TemplateFactory:
  #  namespace = db.onto
    def get_template(self, type : TemplateType):
        return type.value()

    def get_all_template_names(self):
        return TemplateType._member_names_
    
    
    