import sys, os; sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
import numpy as np
from collections import Counter

if __name__ == '__main__':
    n = sys.argv[1]
    dataset = np.load(os.path.expanduser(f'~/ros2_ws/src/imitrob-hri/imitrob-hri/data/artificial_dataset_0{n}.npy'), allow_pickle=True)
    
    tmplts = []
    slctns = []
    strgs = []
    for sample in dataset:
        
        tmplts.append( sample['y']['template'] )
        slctns.append( sample['y']['selections'] )
        strgs.append( sample['y']['storages'] )

    print(f"templates: {Counter(tmplts)}")
    print(f"selections: {Counter(slctns)}")
    print(f"storages: {Counter(strgs)}")

    print("---")
    print(f"y: {sample['y']['template']}, {sample['y']['selections']}, {sample['y']['storages']}")
    print(f"config {sample['config']}")
    print(f"x {sample['x_scene']}")
    print(f"x {sample['x_sentence']}")

    print("---")

    glued_n = 0
    pickable_n = 0
    reachable_n = 0
    stackable_n = 0
    pushable_n = 0
    full_n = 0

    for sample in dataset:
        glued_n += int(sample['x_scene'].selections[0].glued()[1])
        pickable_n += int(sample['x_scene'].selections[0].pickable()[1])
        reachable_n += int(sample['x_scene'].selections[0].reachable()[1])
        stackable_n += int(sample['x_scene'].selections[0].stackable()[1])
        pushable_n += int(sample['x_scene'].selections[0].pushable()[1])
        full_n += int(sample['x_scene'].selections[0].full()[1])

    print(glued_n, pickable_n, reachable_n, stackable_n, pushable_n, full_n)