

from imitrob_hri.data.scene3_def import *
from imitrob_hri.merging_modalities.modality_merger import MMSentence, ModalityMerger
from imitrob_hri.merging_modalities.probs_vector import ProbsVector
from imitrob_hri.merging_modalities.configuration import ConfigurationCrow1 
from imitrob_templates.templates.StackTask import StackTask

# 1. I add sample properties and generated the scene s
s = create_scene_from_fake_data()
# 2. Real configuration:
c = ConfigurationCrow1()
# 3. Make merge
use_magic = 'entropy_add_2'
model = 3


L = {
    'template': ProbsVector([1.0], ['stack'], c=c),
    'selections': ProbsVector([1.0], ['cleaner'], c=c),
    'storages': ProbsVector([1.0], ['crackers'], c=c),
}
G = {
    'template':   ProbsVector([], [], c=c),
    'selections': ProbsVector([], [], c=c),
    'storages':   ProbsVector([], [], c=c),
}


mms = MMSentence(L=L, G=G)
mms.make_conjunction(c)


task = StackTask()

for ob in s.selections:
    for st in s.storages:
        print(f"object {ob.name}, storage {st.name}, {task.is_feasible(ob, st)}")


mm = ModalityMerger(c, use_magic)
mms.M, DEBUGdata = mm.feedforward3(mms.L, mms.G, scene=s, epsilon=c.epsilon, gamma=c.gamma, alpha_penal=c.alpha_penal, model=model, use_magic=use_magic)

print("TEMPLATE")
print(mms.M['template'])
print("SELECTIONS")
print(mms.M['selections'])
print("STORAGES")
print(mms.M['storages'])


