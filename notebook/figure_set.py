import matplotlib.pyplot as plt


def fig_set():
    p = plt.rcParams
    p["figure.figsize"] = 6, 2.5
    p["figure.edgecolor"] = "black"
    p["figure.facecolor"] = "#f9f9f9"
    p["axes.linewidth"] = 1
    p["axes.facecolor"] = "#f9f9f9"
    p["axes.ymargin"] = 0.1
    p["axes.spines.bottom"] = True
    p["axes.spines.left"] = True
    p["axes.spines.right"] = False
    p["axes.spines.top"] = False
    #p["font.sans-serif"] = ["Fira Sans Condensed"]
    p["axes.grid"] = False
    p["grid.color"] = "black"
    p["grid.linewidth"] = 0.1
    p["xtick.bottom"] = True
    p["xtick.top"] = False
    p["xtick.direction"] = "out"
    p["xtick.major.size"] = 5
    p["xtick.major.width"] = 1
    p["xtick.minor.size"] = 3
    p["xtick.minor.width"] = 0.5
    p["xtick.minor.visible"] = True
    p["ytick.left"] = True
    p["ytick.right"] = False
    p["ytick.direction"] = "out"
    p["ytick.major.size"] = 5
    p["ytick.major.width"] = 1
    p["ytick.minor.size"] = 3
    p["ytick.minor.width"] = 0.5
    p["ytick.minor.visible"] = True
    p["lines.linewidth"] = 2
    p["lines.marker"] = "o"
    p["lines.markeredgewidth"] = 1.5
    p["lines.markeredgecolor"] = "auto"
    p["lines.markerfacecolor"] = "white"
    p["lines.markersize"] = 6
