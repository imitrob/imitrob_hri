#!/usr/bin/env python3

# from crow_nlp.nlp_crow.templates.actions.DefineAreaAction import DefineAreaAction
# from crow_nlp.nlp_crow.templates.actions.DemonstrationListAction import DemonstrationListAction
# from crow_nlp.nlp_crow.templates.actions.LearnNewTaskAction import LearnNewTaskAction
# from crow_nlp.nlp_crow.templates.actions.LearnTowerFromDemonstrationAction import LearnTowerFromDemonstrationAction
# from crow_nlp.nlp_crow.templates.tasks.ApplyGlueTask import ApplyGlueTask
# from imitrob_templates.templates.PickTask import PickTask
# from crow_nlp.nlp_crow.templates.tasks.TidyTask import TidyTask
# from imitrob_templates.templates.PointTask import PointTask
# from imitrob_templates.templates.PassTask import PassTask
# from imitrob_templates.templates.PourTask import PourTask
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

# from imitrob_templates.templates.MoveUpTask import MoveUpTask
# from imitrob_templates.templates.PickTask import PickTask
# from imitrob_templates.templates.UnglueTask import UnglueTask
# from imitrob_templates.templates.PointTask import PointTask
# from imitrob_templates.templates.PourTask import PourTask
# from imitrob_templates.templates.PushTask import PushTask
# from imitrob_templates.templates.PutIntoTask import PutIntoTask
# from imitrob_templates.templates.ReleaseTask import ReleaseTask
# from imitrob_templates.templates.StopTask import StopTask
# from imitrob_templates.templates.StackTask import StackTask
# from imitrob_templates.templates.PassTask import PassTask
from imitrob_hri.imitrob_nlp.nlp_utils import template_name_synonyms

from imitrob_templates.templates import imported_classes
from enum import Enum


enum_strs = []
enum_values = []
for cls in imported_classes:
    # <class 'PickTask.PickTask'> to "PICK_TASK"
    enum_strs.append(f"{cls.__name__[:-4].upper()}_TASK")
    enum_values.append(cls)
    
    ## Make second def. for each template 1. "PICK_TASK", 2. "pick"
    # <class 'PutIntoTask.PutIntoTask'> to "PutInto"
    s1 = cls.__name__[:-4]
    # "PutInto" to "Put-Into"
    split_indexs = []
    for n, s_ in enumerate(s1):
        if n>0 and s_.isupper():
            split_indexs.append(n)
    
    split_indexs.reverse()
    for split_index in split_indexs:
        s1 = s1[:split_index] + '-' + s1[split_index:]
    # "Put-Into" to "put-into"
    s1 = s1.lower()

    enum_strs.append(f"{s1}")
    enum_values.append(cls)

TemplateType = Enum('TemplateType', list(zip(enum_strs, enum_values)))

## HERE IS THE OLD SOLUTION
# class TemplateType(Enum):
# mapping from constants to classes
# PICK_TASK = PickTask
# APPLY_GLUE = ApplyGlueTask
# PUT_TASK = PutTask
# TIDY_TASK = TidyTask
# POINT_TASK = PointTask
# PASSME_TASK = PassTask
# POUR_TASK = PourTask
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

class TemplateFactory:
    def get_template(self, type : TemplateType):
        return type.value()

    def get_all_template_names(self):
        return TemplateType._member_names_
    
    def get_template_class_from_str(self, name):
        return getattr(TemplateType,name).value
    


def to_default_name(name, ct='template'):
    name = name.lower()
    name = name.replace("_", " ")
    assert isinstance(name, str), f"name is not string, it is {type(name)}"
    ct_name_synonyms = eval(ct+'_name_synonyms')

    for key in ct_name_synonyms.keys():
        for item in ct_name_synonyms[key]:
            if name == item.lower():
                return ct_name_synonyms[key][0]
    print(f"Exception for {name} not in {ct_name_synonyms}")
    print("returning")

def create_template(template_name, nlp=False):
    template_name = to_default_name(template_name)
    if template_name is None:
        return
    
    return TemplateFactory().get_template_class_from_str(template_name)(nlp=nlp)
    
    ## might be deleted
    # return {
    # 'stop': StopTask,
    # 'release': ReleaseTask,
    # 'move-up': MoveUpTask,
    # 'pick': PickTask,
    # 'point': PointTask,
    # 'push': PushTask,
    # 'unglue': UnglueTask,
    # 'put-into': PutIntoTask,
    # 'pour': PourTask,
    # 'stack': StackTask,
    # 'pass': PassTask
    # }[template_name](nlp=nlp)
    
    
if __name__ == '__main__':
    import time
    tmplts = ['stop',
    'release',
    'move-up',
    'pick',
    'point',
    'push',
    'unglue',
    'put-into',
    'pour',
    'stack',
    'pass']
    
    for tmplt in tmplts:
        t0 = time.perf_counter()
        t = create_template(tmplt)
        print(f"time {tmplt}: {time.perf_counter()-t0}")