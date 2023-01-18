import numpy as np
import matplotlib.pyplot as plt

def draw_unit_circle(ax):
    t = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(t), np.sin(t), "k--")