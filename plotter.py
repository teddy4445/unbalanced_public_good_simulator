# library imports
import os
import numpy as np
import pandas as pd
import seaborn as sns
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
                 "-",
                 markersize=2,
                 color="green",
                 label="$r(t)$")

        plt.plot(model_matrix["t"],
                 np.asarray(model_matrix["p"]),
                 "--",
                 markersize=2,
                 color="red",
                 label="$p(t)$")

        plt.xlabel("Years [t]", fontsize=16)
        plt.ylabel("Population size [1]", fontsize=16)
        #plt.xlim((min(model_matrix["t"]), max(model_matrix["t"])))
        plt.ylim((0, 10000000 * 0.6))
        plt.xlim((0, 50))
        plt.legend()
        plt.grid(alpha=0.25,
                 color="black")
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.tight_layout()
        plt.savefig(save_path, dpi=600)
        plt.close()

    @staticmethod
    def baseline_profit(model_matrix: dict,
                        save_path: str = "baseline_profit.pdf"):

        a = min(model_matrix["r"])
        if a < 0:
            model_matrix["r"] = [val - 1.5*a for val in model_matrix["r"]]

        # TODO: plot the run itself
        ax = plt.subplot(111)

        plt.plot(model_matrix["t"],
                 np.asarray(model_matrix["r"]),
                 "-",
                 markersize=2,
                 color="green",
                 label="$u_r(t)$")

        plt.plot(model_matrix["t"],
                 np.asarray(model_matrix["p"]),
                 "--",
                 markersize=2,
                 color="red",
                 label="$u_p(t)$")

        plt.xlabel("Years [t]", fontsize=16)
        plt.ylabel("Utility [1]", fontsize=16)
        #plt.xlim((min(model_matrix["t"]), max(model_matrix["t"])))
        plt.ylim((0, 4*10000000000))
        plt.xlim((0, 50))
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
                     fmt="--o")
        plt.xlabel("${}$".format(x_label), fontsize=16)
        plt.ylabel(y_label, fontsize=16)
        plt.xticks(x, fontsize=12, rotation=45)
        top_error_point = max(np.asarray(y) + np.asarray(y_err)) * 1.05
        bottom_error_point = min(np.asarray(y) - np.asarray(y_err)) * 1.05
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.ylim((0, 50))
        plt.xlim((min(x), max(x)))
        axes = plt.gca()
        axes.spines[['right', 'top']].set_visible(False)
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
                    vmax=50,
                    annot=False,
                    cmap="coolwarm")
        plt.xlabel(x_label.replace("1", "_r").replace("2", "_p"), fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.savefig(save_path, dpi=600)
        plt.tight_layout()
        plt.close()

        sns.heatmap(df,
                    annot=False,
                    cmap="coolwarm")
        plt.xlabel(x_label.replace("1", "_r").replace("2", "_p"), fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.savefig(save_path.replace(".pdf", "_zoom.pdf"), dpi=600)
        plt.tight_layout()
        plt.close()
