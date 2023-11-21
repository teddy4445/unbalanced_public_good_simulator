# library imports
import os
import numpy as np
import pandas as pd
import seaborn as sns
from dtreeviz.trees import *
import matplotlib.pyplot as plt


# project imports


class Plotter:
    """
    This class responsible to plot all the needed graphs for paper
    """

    # CONSTS #

    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def baseline(model_matrix: dict,
                 save_path: str = "baseline.pdf"):
        # TODO: plot the run itself
        ax = plt.subplot(111)

        plt.plot(model_matrix["t"],
                 np.asarray(model_matrix["r"]),
                 "--",
                 markersize=2,
                 color="green",
                 label="$r(t)$")

        plt.plot(model_matrix["t"],
                 np.asarray(model_matrix["p"]),
                 "--",
                 markersize=2,
                 color="red",
                 label="$p(t)$")

        plt.xlabel("Steps in time", fontsize=16)
        plt.ylabel("Population size", fontsize=16)
        plt.xlim((min(model_matrix["t"]), max(model_matrix["t"])))
        #plt.ylim((0, np.max([max(model_matrix["r"]), max(model_matrix["p"])])))
        plt.ylim((0, 10000))
        plt.legend()
        plt.grid(alpha=0.25,
                 color="black")
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.tight_layout()
        plt.savefig(save_path, dpi=600)
        plt.close()

    @staticmethod
    def sensitivity(x,
                    y,
                    y_err,
                    x_label: str,
                    y_label: str,
                    save_path: str):
        ax = plt.subplot(111)
        plt.errorbar(x,
                     y,
                     y_err,
                     ecolor="blue",
                     color="blue",
                     capsize=3,
                     fmt="-o")
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.xticks(x, fontsize=12)
        top_error_point = max(np.asarray(y) + np.asarray(y_err)) * 1.05
        bottom_error_point = min(np.asarray(y) - np.asarray(y_err)) * 1.05
        plt.yticks([bottom_error_point + 0.1 * i * (abs(top_error_point) + abs(bottom_error_point)) for i in range(11)],
                   fontsize=12)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.grid(alpha=0.25,
                 color="black")
        plt.tight_layout()
        plt.savefig(save_path, dpi=600)
        plt.close()

    @staticmethod
    def heatmap(df: pd.DataFrame,
                x_label: str,
                y_label: str,
                save_path: str):
        sns.heatmap(df,
                    vmin=0,
                    vmax=1,
                    annot=False,
                    cmap="coolwarm")
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.savefig(save_path, dpi=600)
        plt.tight_layout()
        plt.close()

        sns.heatmap(df,
                    vmin=0,
                    annot=False,
                    cmap="coolwarm")
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.savefig(save_path.replace(".pdf", "_zoom.pdf"), dpi=600)
        plt.tight_layout()
        plt.close()

    @staticmethod
    def snail_scatter(df,
                      save_path: str):
        c = df.iloc[:,2]
        max_c = max(c)
        min_c = min(c)
        delta_c = max_c - min_c
        c = [(val-min_c)/delta_c for val in c]
        plt.scatter(df.iloc[:,0],
                    df.iloc[:,1],
                    c=c,
                    marker="o",
                    cmap="coolwarm")
        plt.xlabel("$T_s$")
        plt.ylabel("$T_i$")
        plt.grid(alpha=0.25,
                 color="black")
        ax = plt.gca()
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.tight_layout()
        plt.savefig(save_path,
                    dpi=600)
        plt.close()
