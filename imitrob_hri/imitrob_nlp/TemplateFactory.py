#!/usr/bin/env python3

from imitrob_hri.imitrob_nlp.nlp_utils import template_name_synonyms
from imitrob_templates.templates import imported_classes
from enum import Enum
import numpy as np
import os

enum_strs = []
enum_values = []
for cls in imported_classes:
    ## <class 'PickTask.PickTask'> to "PICK_TASK"
    #enum_strs.append(f"{cls.__name__[:-4].upper()}_TASK")
    #enum_values.append(cls)
    
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
    enum_values.append(getattr(__import__(f'imitrob_templates.templates.{cls.__name__}', globals(), locals(), [cls.__name__], 0), cls.__name__))

    # hack to import class for saving
    globals()[cls.__name__] = getattr(__import__(f'imitrob_templates.templates.{cls.__name__}', globals(), locals(), [cls.__name__], 0), cls.__name__)

TemplateType = Enum('TemplateType', list(zip(enum_strs, enum_values)))

class TemplateFactory:
    def get_template(self, type : TemplateType):
        return type.value()

    def get_all_template_names(self):
        return TemplateType._member_names_
    
    def get_template_class_from_str(self, name):
        return getattr(TemplateType,name).value
    

    def get_all_template_values(self):
        return [getattr(TemplateType, name).value for name in TemplateType._member_names_]

    def get_all_template_detect_functions(self):
        templates = self.get_all_template_values()
        template_detect_functions = [t.detect_fun for t in templates]
        return template_detect_functions


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
    
    ts = []
    for tmplt in tmplts:
        t0 = time.perf_counter()
        t = create_template(tmplt, nlp=False)
        print(f"time {tmplt}: {time.perf_counter()-t0}")
        ts.append(t)
    
    st = [StopTask(nlp=False),
          PassTask(nlp=False),
          StackTask(nlp=False),
          MoveUpTask(nlp=False),
          ]

    np.save('/home/petr/Downloads/test_only', st)