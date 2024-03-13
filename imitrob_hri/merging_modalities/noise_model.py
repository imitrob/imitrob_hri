
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats
import os


class NormalModel():
    ''' Sample Normal Distribution '''
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, size=1):
        ''' normal cropped (0 to 1) '''
        return np.random.normal(self.mu * np.ones((size)), self.sigma * np.ones((size)))


class MixtureModel():
    def __init__(self,  params, factor=1, negative_noise=False):
        self.models = []
        for p in params:
            self.models.append(getattr(sp.stats, p[0])(*p[1]))
        self.factor = factor
        self.negative_noise = negative_noise

    def rvs(self, size):
        submodel_choices = np.random.choice(len(self.models), size=size)
        submodel_samples = [submodel.rvs(size=size) for submodel in self.models]
        rvs = np.choose(submodel_choices, submodel_samples)

        sign = 1
        if self.negative_noise:
            sign = np.random.randint(0,2,size) * 2 - 1
        
        return sign * self.factor * rvs
    
    def __call__(self, size=1):
        return self.rvs(size)

# Static models to be accessible global 
global gesture_det_model, gesture_noise_model, gesture_noise_model2, gesture_noise_model3

# prumer n uniform distr, match threshold 0.25-1.0, 
gesture_det_model = MixtureModel([
    ('norm', (0.70518772634005173, 0.11254289107220866)),
    ('norm', (0.8162473647795723, 0.10747602382933483))
])

proper = MixtureModel([
    ('norm', (0.40518772634005173, 0.11254289107220866)),
    ('norm', (0.5162473647795723, 0.10747602382933483)),
    ('norm', (0.29960713071618444, 0.028368399842165663)),
    ('norm', (0.7337954978857516, 0.06631302990996413)),
    ('norm', (0.5998625653155687, 0.09537271998949513)),
    ('norm', (0.5331632665435483, 0.047239334976977285)),
    ('norm', (0.4599737806474422, 0.04574923462552068)),
    ('norm', (0.7013723305787044, 0.01449694961189483)),
])

gesture_noise_model = MixtureModel([
        ('expon', (1.0167785737536344e-08, 0.005827560175383218)),
        ('exponnorm', (1.768464920150208, 0.15072610225705982, 0.05762642382325739))
    ], factor=0.5, negative_noise=True)

gesture_noise_model2 = MixtureModel([
        ('expon', (1.0167785737536344e-08, 0.005827560175383218)),
        ('exponnorm', (1.768464920150208, 0.15072610225705982, 0.05762642382325739))
    ], factor=1.0, negative_noise=True)

gesture_noise_model3 = MixtureModel([
        ('expon', (1.0167785737536344e-08, 0.005827560175383218)),
        ('exponnorm', (1.768464920150208, 0.15072610225705982, 0.05762642382325739))
    ], factor=2.0, negative_noise=True)

gesture_noise_model4 = MixtureModel([
        ('expon', (1.0167785737536344e-08, 0.005827560175383218)),
        ('exponnorm', (1.768464920150208, 0.15072610225705982, 0.05762642382325739))
    ], factor=3.0, negative_noise=True)


def entropy_tester():
    # newest noises
    noises = {'n0':NormalModel(0.0,0.0), 
             'n1': gesture_noise_model2,
             'n2': NormalModel(0.0,0.2),
             'n3': NormalModel(0.0,0.4),
             'n4': NormalModel(0.0,0.6)}

    nm = NormalModel(0.0, 0.05)
    fig = plt.figure(figsize =(6,3))
    # plt.hist(gesture_det_model(100000), bins=np.linspace(0.001,1,200))
    # plt.hist(noises['n0'](100000), bins=np.linspace(0.001, 1, 100))
    plt.hist(noises['n1'](100000), bins=np.linspace(0.001, 1, 100),zorder=100)
    plt.hist(noises['n2'](100000), bins=np.linspace(0.001, 1, 100))
    plt.hist(noises['n3'](100000), bins=np.linspace(0.001, 1, 100))
    plt.hist(noises['n4'](100000), bins=np.linspace(0.001, 1, 100))
    # plt.hist(noises['n5'](100000), bins=np.linspace(0.001, 1, 100))
    plt.legend(['$n_1^{real}$', '$n_2$', '$n_3$', '$n_4$'])
    #plt.hist(gesture_det_model.rvs(10000), bins=np.linspace(0, 1, 200))
    plt.xlabel("Noise level [-]", fontsize = 15)
    plt.ylabel("Occurrence [-]", fontsize = 15)
    plt.grid()
    plt.axis([0.0, 1.0, 0, 3000])
    
    # plt.yscale('log')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    plt.savefig(f"{os.path.dirname(os.path.abspath(__file__))}/noise_model.eps", dpi=fig.dpi, bbox_inches='tight')

    plt.show()
    for i in range(10):
        print(gesture_noise_model.rvs(1))
    print("---")
    for i in range(10):
        print(gesture_det_model())

if __name__ == '__main__':
    entropy_tester()