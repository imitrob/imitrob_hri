
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.patches as mpatches

def interactive_probs_plotter(cl, cg, action_names, clear_threshold, unsure_threshold, diffs_threshold, mo):
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
        ax.fill_between([-0.5,2.5], ret.max_prob, ret.max_prob-diffs_threshold)
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
    ret = action.match(cl, cg)
    do_the_colors_and_texts(ax, ret)
    
    # Create 6 axes for 6 sliders
    axp1 = plt.axes([0.25, 0.3, 0.65, 0.03])
    axp2 = plt.axes([0.25, 0.25, 0.65, 0.03])
    axp3 = plt.axes([0.25, 0.2, 0.65, 0.03])
    axp4 = plt.axes([0.25, 0.15, 0.65, 0.03])
    axp5 = plt.axes([0.25, 0.1, 0.65, 0.03])
    axp6 = plt.axes([0.25, 0.05, 0.65, 0.03])

    s1 = Slider(axp1, f"{action_names[0]} (L)", 0.0, 1.0, cl[0])
    s2 = Slider(axp2, f"{action_names[1]} (L)", 0.0, 1.0, cl[1])
    s3 = Slider(axp3, f"{action_names[2]} (L)", 0.0, 1.0, cl[2])
    s4 = Slider(axp4, f"{action_names[0]} (G)", 0.0, 1.0, cg[0])
    s5 = Slider(axp5, f"{action_names[1]} (G)", 0.0, 1.0, cg[1])
    s6 = Slider(axp6, f"{action_names[2]} (G)", 0.0, 1.0, cg[2])
    
    # Create function to be called when slider value is changed    
    def update(val):
        cl = [s1.val, s2.val, s3.val]
        cg = [s4.val, s5.val, s6.val]
        action = mo
        ret = action.match(cl, cg)
        ax.cla()
        do_the_colors_and_texts(ax, ret)
    
    # Call update function when slider value is changed
    s1.on_changed(update)
    s2.on_changed(update)
    s3.on_changed(update)
    s4.on_changed(update)
    s5.on_changed(update)
    s6.on_changed(update)

    # Create axes for reset button and create button
    resetax = plt.axes([0.8, 0.0, 0.1, 0.04])
    button = Button(resetax, 'Reset', color='gold',
                    hovercolor='skyblue')

    funmodeax = plt.axes([0.2, 0.0, 0.1, 0.04])
    funmodebutton = Button(funmodeax, 'a*b', color='gold',
                    hovercolor='skyblue')
    funmode2ax = plt.axes([0.3, 0.0, 0.1, 0.04])
    funmode2button = Button(funmode2ax, '|a+b|/2', color='gold',
                    hovercolor='skyblue')

    # Create a function resetSlider to set slider to
    # initial values when Reset button is clicked
    
    def resetSlider(event):
        s1.reset()
        s2.reset()
        s3.reset()        
        s4.reset()
        s5.reset()
        s6.reset()

    def funmodebutton_(event):
        mo.fun = 'mul'
    def funmode2button_(event):
        mo.fun = 'add_2'
    
    # Call resetSlider function when clicked on reset button
    button.on_clicked(resetSlider)
    funmodebutton.on_clicked(funmodebutton_)
    funmode2button.on_clicked(funmode2button_)
    
    # Display graph
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
