import matplotlib.pyplot as plt
import numpy as np


def display_graph(y_list, label_list, color_list):
    length = len(y_list[0])
    x = np.arange(0, length)
    line_list = None
    for idx in range(len(y_list)):
        line = plt.plot(x, y_list[idx], label=label_list[idx], color=color_list[idx])
        if idx == 0:
            line_list = line
        else:
            line_list += line
    plt.legend(line_list, label_list)
    plt.pause(1)
    plt.show(block=False)

