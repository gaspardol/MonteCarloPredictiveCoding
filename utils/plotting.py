from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib import pylab
from matplotlib.ticker import StrMethodFormatter
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as patches
from matplotlib.lines import Line2D

import numpy as np


def setup_fig(zero=False, square=True):
    params = {'legend.fontsize': 14,
              'figure.figsize': (4., 4.),
              'axes.labelsize': 16,
              'axes.titlesize': 18,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14}
    pylab.rcParams.update(params)
    if zero is False:
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))  # No decimal places
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))  # 2 decimal places

    # Customizing the spines
    if not square:
        ax = plt.gca()
        ax.spines['right'].set_visible(False)  # Hide the right spine
        ax.spines['top'].set_visible(False)    # Hide the top spine




def generate_video(imgs, show=False, save=False, title="", file_name="movie"):

    fig = plt.figure()
    plt.title(title)
    frames = []
    
    for img in imgs:
        plt.axis('off')
        frames.append([plt.imshow(img, animated=True, cmap=cm.Greys_r)]) #, cmap=cm.Greys_r
    ani = animation.ArtistAnimation(fig, frames, interval=5, blit=True, repeat_delay=1000)
    if show:
        plt.show()
    if save:
        print("saving video as GIF")
        ani.save('figures/'+file_name+'.gif', fps = 500)

def proba_to_coordinate(probs):
    class_polar = np.arange(0., 10.)*2*np.pi/10
    class_x = np.cos(class_polar).reshape((1,-1))
    class_y = np.sin(class_polar).reshape((1,-1))
    x = (probs*class_x).sum(1)
    y = (probs*class_y).sum(1)
    return (x, y), (class_x.squeeze(), class_y.squeeze())


class HandlerLinePatch(HandlerPatch):
    def __init__(self, linehandle=None, **kw):
        HandlerPatch.__init__(self, **kw)
        self.linehandle=linehandle
    
    def create_artists(self, legend, orig_handle, 
                       xdescent, ydescent, width, 
                       height, fontsize, trans):
        p = super().create_artists(legend, orig_handle, 
                                   xdescent, ydescent,
                                   width, height, fontsize, 
                                   trans)
        line = Line2D([0,width],[height/2.,height/2.])
        if self.linehandle is None:
            line.set_linestyle('-')
            line._color = orig_handle._edgecolor
        else:
            self.update_prop(line, self.linehandle, legend)
            line.set_drawstyle('default')
            line.set_marker('')
            line.set_transform(trans)
        return [p[0],line]

def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = patches.FancyArrow(width/2., height/2., width/5., 0, 
                           length_includes_head=True, width=0, 
                           head_width=height, head_length=height, 
                           overhang=0.2)
    return p

def add_arrow(line, ax, position=None, direction='right', color=None, label='', dx=1):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    color:      if None, line color is taken.
    label:      label for arrow
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + dx
    else:
        end_ind = start_ind - dx
    
    dx = xdata[end_ind] - xdata[start_ind]
    dy = ydata[end_ind] - ydata[start_ind]
    size = abs(dx) * 5.
    x = xdata[start_ind] + (np.sign(dx) * size/2.)
    y = ydata[start_ind] + (np.sign(dy) * size/2.)

    arrow = patches.FancyArrow(x, y, dx, dy, color=color, width=0, 
                               head_width=size, head_length=size, 
                               label=label,length_includes_head=True, 
                               overhang=0.3, zorder=10)
    ax.add_patch(arrow)

def plot_line_with_arrow(x,y,ax=None,label='',position=None,dx=10,**kw):
    if ax is None:
        ax = plt.gca()
    line = ax.plot(x,y,**kw)[0]
    add_arrow(line, ax, label=label, position=position, dx=dx)
    return line