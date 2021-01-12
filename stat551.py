# practice problems for stat 551

import stemgraphic
import numpy as np
#import _tkinter
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

#mpl.use('TkAgg')

# simple test to see if plotting works
# not working
# more info here: https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.plot.html
#def check_plot():
    #fig, ax = plt.plot(x,range(10))
    #fig.savefig("check_plot.pdf")

# create a stem and leaf plot and save as a pdf
def stem_leaf():
    data = [56, 74, 25, 68, 67, 68, 90, 78, 85, 83, 46, 93]
    fig, ax = stemgraphic.stem_graphic(data, scale = 10)
    fig.savefig("stem_leaf.pdf")

def histogram():
    data = np.array([56, 74, 25, 68, 67, 68, 90, 78, 85, 83, 46, 93])
    plt.hist(data, bins = [20,40,60,80,100])
    #plt.plot(data)
    canvas = plt.get_current_fig_manager().canvas
    agg = canvas.switch_backends(FigureCanvasAgg)
    agg.draw()
    s, (width, height) = agg.print_to_buffer()
    X = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    im = Image.frombytes("RGBA", (width, height), s)
    #plt.hist(data, bins = [20,40,60,80,100])
    #plt.title("histogram")
    #plt.show()
    im.show()


#check_plot()
#stem_leaf()
histogram()
