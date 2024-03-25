from configuration import Configuration
from modality_merger import ProbsVector, SingleTypeModalityMerger, ModalityMerger, MultiProbsVector
from utils import *
import os
import sys
SCRIPT_DIR = os.path.dirname(__file__)
MODULE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.append(MODULE_DIR)
import numpy as np
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    use_magic = sys.argv[2] if len(sys.argv) > 2 else 'entropy_add_2'
    model = sys.argv[3] if len(sys.argv) > 3 else 'M3'

    dataset = np.load(os.path.join(MODULE_DIR, 'data/saves/artificial_dataset2_{dataset_name}.npy'), allow_pickle=True)

    ''' Set configuration '''
    c = dataset[0]['config']
    nsamples = len(dataset)

    acc_acc = []
    for iter in trange(0, 10, desc='Iter'):
        np.random.seed(np.random.randint(0, int(2**32)))

        acc = 0
        conflict = 0
        for n, sample in tenumerate(dataset, total=nsamples, desc='Sample'):
            s = sample['x_sentence']
            s.make_conjunction(c)

            gt_l = s.L["template"].p[np.where(np.array(s.L["template"].names) == sample["y"]["template"])]
            gt_g = s.G["template"].p[np.where(np.array(s.G["template"].names) == sample["y"]["template"])]
            if gt_l == 0 or gt_g == 0:
                conflict += 1

            scene = sample['x_scene']
            mm = ModalityMerger(c, "add_2")
            #s.M = mm.feedforward2(s.L, s.G, scene)
            s.M, DEBUGdata = mm.feedforward3(s.L, s.G, scene, epsilon=c.epsilon, gamma=c.gamma, alpha_penal=c.alpha_penal, model=model, use_magic=use_magic)

            if s.check_merged(sample['y'], c, printer=False):
                acc +=1

        acc_acc.append(acc / nsamples * 100)
        tqdm.write(f"acc at iter {iter}: {acc_acc[-1]}%")
        tqdm.write(f"conflict at iter {iter}: {conflict/nsamples*100}%")

    print(f"All accuracies: {acc_acc}")

    acc_acc = np.array(acc_acc)
    print(f"Final acc: {acc_acc.mean():0.2f} +- {acc_acc.std():0.2f}\nmin: {acc_acc.min():0.2f}\nmax: {acc_acc.max():0.2f}")
