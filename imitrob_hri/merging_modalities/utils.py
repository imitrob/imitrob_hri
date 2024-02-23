
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.patches as mpatches
import numpy as np
import os

def interactive_probs_plotter(cl, cg, action_names, clear_threshold, unsure_threshold, diffs_threshold, mo, save=False, save_file='somegoodname'):
    '''
    Parameters:
        cl (Float[]): Probability vector Language
        cg (Float[]): Probability vector Gestures
        action_names (String[])
        clear_threshold (Float): Norm 0-1, if prob > clear_threshold then we are clear about action
        unsure_threshold (Float): Norm 0-1, if unsure_threshold < prob < clear_threshold then we are unsure about action
        diffs_threshold (Float): Norm 0-1, safety range for activation, no other clear actions need to be in this range for the action to be considered as activated
        # FIX: TMP:
        mo (SingleTypeModalityMerger) - object
    '''
    def do_the_colors_and_texts(ax, ret):
        colors = []
        for action in action_names:
            if action == ret.activated:
                colors.append('yellow')
            elif action in ret.clear:
                colors.append('orange')
            elif action in ret.unsure:
                colors.append('blue')
            elif action in ret.negative:
                colors.append('red')
            else:
                colors.append('black')
        
        ax.axhline(y=clear_threshold, color='orange', linestyle='--')
        ax.text(2.55, clear_threshold, 'clear threshold', horizontalalignment='right',      verticalalignment='top', fontsize=6)
        ax.axhline(y=unsure_threshold, color='b', linestyle='--')
        ax.text(2.55, unsure_threshold, 'unsure threshold', horizontalalignment='right',      verticalalignment='top', fontsize=6)
        ax.fill_between([0.,len(action_names)-0.5], ret.max_prob, ret.max_prob-diffs_threshold)
        ax.bar(x, ret.p, color=colors,
            edgecolor="black")
        ax.text(1.0, ret.max_prob, 'no-zone', horizontalalignment='left',      verticalalignment='top', fontsize=6)
        activated_patch = mpatches.Patch(color='yellow', label='Activated')
        clear_patch = mpatches.Patch(color='orange', label='Clear')
        unsure_patch = mpatches.Patch(color='blue', label='Unsure')
        negative_patch = mpatches.Patch(color='red', label='Negative')
        ax.legend(handles=[activated_patch,clear_patch,unsure_patch,negative_patch], prop={'size': 6})
        ax.text(0.8, 1.05, f"Conclusion: {ret.conclude()}", horizontalalignment='left', verticalalignment='top', fontsize=6, transform=ax.transAxes)
    x = action_names
    # Create a subplot
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.4)

    action = mo

    if cg is None: # Single ProbsVector plot? 
        ret = mo._get_single_probs_vector(cl, action_names)
    else:
        ret = action.match(cl, cg)
    do_the_colors_and_texts(ax, ret)
    
    axp1 = plt.axes([0.2, 0.2, 0.65, 0.03])
    s1 = Slider(axp1, f"{action_names[0]} (L)", 0.0, 1.0, cl[0])
    if cg is not None: # Multi ProbsVector plot? 
        axp2 = plt.axes([0.2, 0.15, 0.65, 0.03])
        s2 = Slider(axp2, f"{action_names[0]} (G)", 0.0, 1.0, cg[0])
    
    global interactive_probs_plotter__item_n
    interactive_probs_plotter__item_n = 0
    # Create function to be called when slider value is changed    
    def update(val):
        cl[interactive_probs_plotter__item_n] = s1.val
        action = mo
        
        if cg is not None: # multi probs vector plot ?
            cg[interactive_probs_plotter__item_n] = s2.val
        
        if cg is None: # single probs vector plot ?
            ret = mo._get_single_probs_vector(cl, action_names)
        else:
            ret = action.match(cl, cg)
        ax.cla()
        do_the_colors_and_texts(ax, ret)
    
    # Call update function when slider value is changed
    s1.on_changed(update)
    if cg is not None: # Multi ProbsVector plot? 
        s2.on_changed(update)
    
    # Create axes for reset button and create button
    resetax = plt.axes([0.8, 0.0, 0.1, 0.04])
    button = Button(resetax, 'Reset', color='gold',
                    hovercolor='skyblue')

    funmodeax =  plt.axes([0.2, 0.0, 0.1, 0.04])
    funmodebutton = Button(funmodeax, 'a*b', color='gold', 
                    hovercolor='skyblue')
    funmode2ax = plt.axes([0.3, 0.0, 0.1, 0.04])
    funmode2button = Button(funmode2ax, '|a+b|/2', color='gold',
                    hovercolor='skyblue')
    funmode3ax = plt.axes([0.4, 0.0, 0.1, 0.04])
    funmode3button = Button(funmode3ax, 'max', color='gold',
                    hovercolor='skyblue')
    funmode4ax = plt.axes([0.5, 0.0, 0.1, 0.04])
    funmode4button = Button(funmode4ax, 'a*b (ent)', color='gold',
                    hovercolor='skyblue')
    funmode5ax = plt.axes([0.6, 0.0, 0.1, 0.04])
    funmode5button = Button(funmode5ax, 'a+b (ent)', color='gold',
                    hovercolor='skyblue')

    # I don't know how to pass argument at the moment
    def assign_set(nn):
        global interactive_probs_plotter__item_n
        interactive_probs_plotter__item_n = nn

        s1.eventson = False
        s1.set_val(cl[interactive_probs_plotter__item_n])
        fig.canvas.draw()
        s1.eventson = True
        if cg is not None: # multi probs vector plot ?
            s2.eventson = False
            s2.set_val(cg[interactive_probs_plotter__item_n])
            fig.canvas.draw()
            s2.eventson = True

    def assign_set_0(event):
        assign_set(0)
    def assign_set_1(event):
        assign_set(1)
    def assign_set_2(event):
        assign_set(2)
    def assign_set_3(event):
        assign_set(3)
    def assign_set_4(event):
        assign_set(4)
    def assign_set_5(event):
        assign_set(5)
    def assign_set_6(event):
        assign_set(6)
    def assign_set_7(event):
        assign_set(7)
    def assign_set_8(event):
        assign_set(8)

    switchingButtons = []
    n = len(cl)
    for i in range(n):
        
        tester = plt.axes([(1.0/n*0.78)*i+0.12, 0.35, (1.0/n)*0.78, 0.04])
        button = Button(tester, action_names[i], color='gold',              hovercolor='skyblue')
        fun = eval('assign_set_'+str(i))
        button.on_clicked(fun)
        switchingButtons.append(button)


    # Create a function resetSlider to set slider to
    # initial values when Reset button is clicked
    
    def resetSlider(event):
        s1.reset()
        if cg is None: # Single ProbsVector plot? 
            s2.reset()
        
    def funmodebutton_(event):
        mo.fun = 'mul'
    def funmode2button_(event):
        mo.fun = 'add_2'
    def funmode3button_(event):
        raise Exception("TODO")
        mo.fun = 'baseline'
    def funmode4button_(event):
        raise Exception("TODO")
        mo.fun = 'entropy'
    def funmode5button_(event):
        raise Exception("TODO")
        mo.fun = 'entropy_add_2'

    # Call resetSlider function when clicked on reset button
    button.on_clicked(resetSlider)
    funmodebutton.on_clicked(funmodebutton_)
    funmode2button.on_clicked(funmode2button_)
    funmode3button.on_clicked(funmode3button_)
    funmode4button.on_clicked(funmode4button_)
    funmode5button.on_clicked(funmode5button_)
    
    # Display graph
    if save:
        plt.savefig(f"/home/petr/Downloads/{save_file}.eps")
        plt.savefig(f"/home/petr/Downloads/{save_file}.png")
    plt.show()

    
class cc:
    H = '\033[95m'
    OK = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    W = '\033[93m'
    F = '\033[91m'
    E = '\033[0m'
    B = '\033[1m'
    U = '\033[4m'


def entropy(v):
    return -np.sum([x * np.log2(x) for x in v])

def safe_entropy(v):
    v = np.asarray(v, dtype=float)
    v += np.finfo(float).eps  # add epsilon against x / 0
    v = v / np.sum(v) # normalize
    return entropy(v)

def normalized_entropy(v):
    if len(v) == 1: return v[0]
    return safe_entropy(v) / np.log2(len(v))

def cross_entropy(v, q):
    return -np.sum([vx * np.log2(qx) for vx, qx in zip(v, q)])

def safe_cross_entropy(v, q):
    assert len(v) == len(q)
    v, q = np.asarray(v, dtype=float), np.asarray(q, dtype=float)
    v += np.finfo(float).eps  # add epsilon against x / 0
    q += np.finfo(float).eps  # add epsilon against x / 0
    v = v / np.sum(v) # normalize
    q = q / np.sum(q) # normalize
    return cross_entropy(v, q)

def normalized_cross_entropy(v, q):
    return safe_cross_entropy(v, q) / np.log2(len(v))

def  diagonal_cross_entropy(v):
    if len(v) == 1: return v[0]
    return [normalized_cross_entropy(np.eye(len(v))[i], v) for i in range(len(v))]


def singlehistplot_customized(data, filename, labels=['baseline','M1', 'M2', 'M3'], 
                              xticks=['D1','D2','D3','D4','D5'], xlbl='', ylbl='Accuracy [%]',
                              bottom=0, plot=False, savefig=True, figsize=(12,6), title=""):
    ''' Plot histogram plot: Used at MM paper results
    Parameters:
        data (Float[][]): (bars, series?)
        filename (String): Saved file name (without extension)
        labels (String[]): Series names
        xticks (String[]): Histogram names
        ylbl (String): Y axis label
        bottom (Float): Bar numbering from this number
            (e.g. show accuracy data from 80% to 100%: bottom = 80)
        plot (Bool): plt.show()
        save (Bool): savefig
    '''
    
    print("Data", data)
    shape = data.shape
    xl = shape[0]
    yl = shape[1]

    # set width of bar
    barWidth = 0.1
    fig = plt.figure(figsize=figsize)

    # bars
    brs = [np.arange(xl)]
    for i in range(1,yl):
        brs.append( [x + barWidth for x in brs[i-1]])
    plt.grid(axis='y')
    
    colors = iter([plt.cm.tab20(i) for i in range(20)])

    arange = list(range(-1, -len(brs)-1,-1))
    for n in arange:
        br, d_, l_ = brs[n], data[:,n], labels[n]
        plt.bar(br, d_-bottom, color =next(colors), width = barWidth,
                edgecolor ='grey', label =l_, bottom=bottom)

    plt.ylim(top = 100)
    # Adding Xticks
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    if title != "":
        plt.title(title)
    if xlbl != '':
        plt.xlabel(xlbl, fontsize = 15)
    plt.ylabel(ylbl, fontsize = 15)
    plt.xticks([r + barWidth for r in range(xl)],
            xticks)
    plt.xticks(rotation=90)

    plt.legend(loc='lower left')
    
    if savefig:
        plt.savefig(f"{os.path.expanduser(os.path.dirname(os.path.abspath(__file__)))}/../data/pictures/{filename}.eps", dpi=fig.dpi, bbox_inches='tight')
        plt.savefig(f"{os.path.expanduser(os.path.dirname(os.path.abspath(__file__)))}/../data/pictures/{filename}.png", dpi=fig.dpi, bbox_inches='tight')
    if plot:
        plt.show()


class ForDebugOnly():
    def __init__(self):
        self.arity = {'template': [],
                      'selections': [],
                      'storages': [],}
        self.properties = {'template': [],
                      'selections': [],
                      'storages': [],}
        
fdo = ForDebugOnly()