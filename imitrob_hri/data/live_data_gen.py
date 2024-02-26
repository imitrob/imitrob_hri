import uuid
from abc import ABCMeta, abstractproperty

import numpy as np
import pandas as pd
from networkx import constraint

RNG = np.random.default_rng(uuid.uuid4().int)
# RNG = np.random.default_rng(12345)


def translate_marker(marker):
    if "+" in marker:
        return True
    elif "-" in marker or "." in marker:
        return False
    elif "*" in marker:
        return None


class Entity(metaclass=ABCMeta):
    properties: set
    NICE_TO_BE_TRUE = []
    NICE_TO_BE_FALSE = []

    def __init__(self, name, record):
        self.type = None
        self.name = name
        cls = self.__class__
        for k, v in record.items():
            if k in ["has", "property"]:
                continue
            cls.properties.add(k)
            setattr(self, k, translate_marker(v))

    def __getattribute__(self, key):
        v = super().__getattribute__(key)
        if not key.startswith("_") and v is None and hasattr(self, f"_proxy_{key}"):
            v = getattr(self, f"_proxy_{key}")
        return v

    def decide(self):
        if self.name == "":
            raise ValueError("Don't run 'decide' on action's proxy objects!")
        d = {**self.__dict__}
        for k, v in d.items():
            if k.startswith("_"):
                continue
            if v is None:
                setattr(self, f"_proxy_{k}", bool(RNG.integers(0, 2)))

    def decide_nicely(self):
        if self.name == "":
            raise ValueError("Don't run 'decide' on action's proxy objects!")
        d = {**self.__dict__}
        for k, v in d.items():
            if k.startswith("_"):
                continue
            if v is None:
                if k in self.NICE_TO_BE_TRUE:
                    decided_value = True
                elif k in self.NICE_TO_BE_FALSE:
                    decided_value = False
                else:
                    decided_value = bool(RNG.integers(0, 2))

                setattr(self, f"_proxy_{k}", decided_value)

    def decide_ugly(self):
        if self.name == "":
            raise ValueError("Don't run 'decide' on action's proxy objects!")
        d = {**self.__dict__}
        for k, v in d.items():
            if k.startswith("_"):
                continue
            if v is None:
                if k in self.NICE_TO_BE_TRUE:
                    decided_value = False
                elif k in self.NICE_TO_BE_FALSE:
                    decided_value = True
                else:
                    decided_value = bool(RNG.integers(0, 2))

                setattr(self, f"_proxy_{k}", decided_value)

    def has_same_properties(self, other):
        if self.type != other.type:
            return False
        for self_prop, self_prop_value in self.constraints.items():
            if self_prop_value is None:
                continue
            if other.constraints.get(self_prop) != self_prop_value:
                return False
        return True

    @property
    def constraints(self):
        prop_fields = self.__class__.properties
        return {k: getattr(self, k) for k in prop_fields}

    @property
    def df_constraints(self):
        return pd.DataFrame.from_dict(self.constraints, orient="index", columns=[self.name])


class Object(Entity):
    properties = set()
    NICE_TO_BE_TRUE = ["reachable", "pushable", "full_liquid"]
    NICE_TO_BE_FALSE = ["glued", "full_stack"]

    def __init__(self, name, record):
        super().__init__(name, record)
        self.type = "object"


class Storage(Entity):
    properties = set()
    NICE_TO_BE_TRUE = ["reachable"]
    NICE_TO_BE_FALSE = ["full_liquid", "full_stack", "full_container"]

    def __init__(self, name, record):
        super().__init__(name, record)
        self.type = "storage"


class Action:

    def __init__(self, name, record):
        self.name = name
        object_record = record["object"]
        storage_record = record["storage"]
        self.has_object = "+" in object_record.loc["has"]
        self.has_storage = "+" in storage_record.loc["has"]
        self.arity = int(self.has_object) + int(self.has_storage)

        if self.has_object:
            self.object = Object("", object_record)
        else:
            self.object = None

        if self.has_storage:
            self.storage = Storage("", storage_record)
        else:
            self.storage = None

    def is_object_feasible(self, obj):
        if self.has_object:
            if obj is None:
                return False
            else:
                return self.object.has_same_properties(obj)
        elif obj is None:
            return True
        else:
            return False

    def is_storage_feasible(self, storage):
        if self.has_storage:
            if storage is None:
                return False
            else:
                return self.storage.has_same_properties(storage)
        elif storage is None:
            return True
        else:
            return False

    def object_constraints(self):
        if self.has_object:
            return self.object.constraints

    def storage_constraints(self):
        if self.has_storage:
            return self.storage.constraints

    def object_df_constraints(self):
        if self.has_object:
            df = self.object.df_constraints
            df.columns = [f"{self.name} object"]
            return df

    def storage_df_constraints(self):
        if self.has_storage:
            df = self.storage.df_constraints
            df.columns = [f"{self.name} storage"]
            return df


class DataGen:

    def __init__(self, table_file_path="src/imitrob_hri/imitrob_hri/data/action_obj_props_HRI.ods") -> None:
        self.object_table = pd.read_excel(table_file_path, engine="odf", sheet_name="objects", index_col=0)
        self.storage_table = pd.read_excel(table_file_path, engine="odf", sheet_name="storages", index_col=0)
        self.action_table = pd.read_excel(table_file_path, engine="odf", sheet_name="actions", index_col=0, header=[0, 1])

        self.actions = {}
        self.objects = {}
        self.storages = {}

        for name, record in self.action_table.iterrows():
            a = Action(name, record)
            self.actions[name] = a

        for name, record in self.object_table.iterrows():
            o = Object(name, record)
            self.objects[name] = o

        for name, record in self.storage_table.iterrows():
            s = Storage(name, record)
            self.storages[name] = s

    def get_entity(self, entity_name):
        if entity_name in self.objects:
            return self.objects[entity_name]
        elif entity_name in self.storages:
            return self.storages[entity_name]
        else:
            return None

    def get_random_entity(self):
        return RNG.choice(list(self.objects.values()) + list(self.storages.values()))

    def get_anti_arity_actions(self, arity):
        anti_act = []
        for k, v in self.actions.items():
            if v.arity != arity:
                anti_act.append(k)
        return anti_act

    def _get_anti_prop(self, action):
        if action.arity == 0:
            return None

        # obj_props = action.object.properties
        # anti_obj = []
        # for other_action_name, other_action in self.objects.items():
        #     if not other_action.has_object:
        #         continue
        #     if
        #     if prop not in other_action.properties:
        #         anti_obj.append(other_action_name)
        # return anti_obj

    def get_object_set_antiset(self, action_name):
        action = self.actions[action_name]

        feasible_objects = []
        nonfeasible_objects = []
        for o in self.objects.values():
            if action.is_object_feasible(o):
                feasible_objects.append(o)
            else:
                nonfeasible_objects.append(o)

        return feasible_objects, nonfeasible_objects

    def get_storage_set_antiset(self, action_name):
        action = self.actions[action_name]

        feasible_storages = []
        nonfeasible_storages = []
        for s in self.storages.values():
            if action.is_storage_feasible(s):
                feasible_storages.append(s)
            else:
                nonfeasible_storages.append(s)

        return feasible_storages, nonfeasible_storages

    # def get_anitset(self, action_name):
    #     action = self.actions[action_name]

    #     # anti arity actions
    #     anti_arity = self._get_anti_arity(action.arity)

    #     # anti property actions


    #     return anti_arity

    def find_feasible_entities_for_action(self, action_name):
        action = self.actions[action_name]
        feasible_object = None
        if action.has_object:
            feasible_objects, _ = self.get_object_set_antiset(action_name)
            if feasible_objects == []:
                raise ValueError("No feasible objects for action {}".format(action_name))
            feasible_object = RNG.choice(feasible_objects)

        feasible_storage = None
        if action.has_storage:
            feasible_storages, _ = self.get_storage_set_antiset(action_name)
            if feasible_storages == []:
                raise ValueError("No feasible storages for action {}".format(action_name))
            feasible_storage = RNG.choice(feasible_storages)

        return feasible_object, feasible_storage

    def decide_objects(self):
        for o in self.objects.values():
            o.decide()

    def decide_storages(self):
        for s in self.storages.values():
            s.decide()

    def decide_objects_nicely(self):
        for o in self.objects.values():
            o.decide_nicely()

    def decide_storages_nicely(self):
        for s in self.storages.values():
            s.decide_nicely()

    def decide_all(self):
        self.decide_objects()
        self.decide_storages()

    def decide_all_nicely(self):
        self.decide_objects_nicely()
        self.decide_storages_nicely()

    def get_random_action(self, allowed_arity=[0, 1, 2]):
        allowed_actions = []
        for k, v in self.actions.items():
            if v.arity in allowed_arity:
                allowed_actions.append(k)
        choice = RNG.choice(allowed_actions)
        return choice

    def print_action(self, action_name):
        action = self.actions[action_name]
        print(f"'{action.name}' action information:")
        print(f"arity: {action.arity}")
        if action.has_object:
            print(f"Object constraints:\n{action.object_constraints()}")
        else:
            print("action has no object")

        if action.has_storage:
            print(f"Storage constraints:\n{action.storage_constraints()}")
        else:
            print("action has no storage")


def print_objects_on_scene(objects_on_scene):
    print("Objects on scene:")
    for obj in objects_on_scene:
        print(f"{obj.name}:")
        print(f"\t{obj.constraints}")


if __name__ == "__main__":
    TABLE_FILE_PATH = "src/imitrob_hri/imitrob_hri/data/action_obj_props_HRI.ods"
    OTHER_ACTION_FEASIBLE = False
    OBJECTS_ON_SCENE_MAX = 5
    SETS_PER_DECIDABILITY = 1

    data_gen = DataGen(TABLE_FILE_PATH)

    for DECIDABLE_ON_PROPS_ONLY in [True, False]:
        for ds_iter in range(SETS_PER_DECIDABILITY):
            # sel_action = "put_into"
            target_action_name = data_gen.get_random_action(allowed_arity=[1, 2])

            target_action = data_gen.actions[target_action_name]
            data_gen.print_action(target_action_name)

            # data_gen.decide_all()
            data_gen.decide_all_nicely()

            target_object, target_storage = data_gen.find_feasible_entities_for_action(target_action_name)
            # print(o.df_constraints if o is not None else "> No object <")
            # print(s.df_constraints if s is not None else "> No storage <")

            other_arity_allowed = [target_action.arity] if DECIDABLE_ON_PROPS_ONLY else [ai for ai in range(3) if ai != target_action.arity]
            for i in range(100):
                other_action_name = data_gen.get_random_action(allowed_arity=other_arity_allowed)
                if other_action_name == target_action_name:  # if same action picked, try again
                    continue
                other_action = data_gen.actions[other_action_name]
                if DECIDABLE_ON_PROPS_ONLY:
                    if other_action.is_object_feasible(target_object) and other_action.is_storage_feasible(target_storage):
                        continue  # if other action is feasible on the same targets, try again
                break  # good action found
            else:
                raise ValueError("No other action found within time limit!")

            already_decided_entities = []
            if target_object is not None:
                already_decided_entities.append(target_object.name)
            if target_storage is not None:
                already_decided_entities.append(target_storage.name)

            other_object, other_storage = None, None
            if OTHER_ACTION_FEASIBLE:
                for i in range(100):
                    other_object, other_storage = data_gen.find_feasible_entities_for_action(other_action_name)
                    if target_action.is_object_feasible(other_object) and target_action.is_storage_feasible(other_storage):
                        continue  # if other action is feasible on the same targets, try again

                    if other_object is not None:
                        already_decided_entities.append(other_object.name)
                    if other_storage is not None:
                        already_decided_entities.append(other_storage.name)
                    break  # good action found
                else:
                    raise ValueError("No other action targets found within time limit!")

            for e in [*data_gen.objects.values(), *data_gen.storages.values()]:
                if e.name in already_decided_entities:
                    continue
                e.decide_ugly()

            if not OTHER_ACTION_FEASIBLE:
                for i in range(100):
                    other_object = data_gen.get_random_entity()
                    if target_action.is_object_feasible(other_object):
                        continue

                    already_decided_entities.append(other_object.name)
                    break  # good action found
                else:
                    raise ValueError("No other object found within time limit!")

            objects_on_scene = [data_gen.get_entity(e) for e in already_decided_entities]

            for rnd_obj_iters in range(OBJECTS_ON_SCENE_MAX - len(objects_on_scene)):
                rnd_obj = data_gen.get_random_entity()
                if rnd_obj.name in already_decided_entities:
                    continue

                objects_on_scene.append(rnd_obj)

            print_objects_on_scene(objects_on_scene)
            print(f"Correct sentence: {target_action_name} {target_object.name if target_object is not None else ''} {'to ' + target_storage.name if target_storage is not None else ''}")
            print(f"Incorrect action: {other_action_name} {target_object.name if target_object is not None else ''} {'to ' + target_storage.name if target_storage is not None else ''}")
            print(f"Incorrect object: {target_action_name} {other_object.name if other_object is not None else ''} {'to ' + target_storage.name if target_storage is not None else ''}")
