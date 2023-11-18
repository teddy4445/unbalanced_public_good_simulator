# library imports
import time
import pickle
import oct2py
import random
from dtreeviz.trees import *
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# project imports
from plotter import Plotter
from mat_file_loader import MatFileLoader

# initialize for the system
oct2py.octave.addpath(os.path.dirname(__file__))


# places classification
def to_classes(row):
    try:
        if row["T_i"] + row["T_i"] == 0:
            return 0
        answer = int(round(row["S"] / (row["T_i"] + row["T_i"]), 1) * 10)
        if answer > 2:
            return 2
        else:
            return answer
    except:
        return 0


class Main:
    """
    The main class of the project
    """

    # CONSTS #
    RESULTS_FOLDER = "results"

    OCTAVE_BASED_SCRIPT_NAME = "model_solver.txt"
    OCTAVE_RUN_SCRIPT_NAME = "model_solver.m"
    OCTAVE_RUN_RESULT_NAME = "model_answer.mat"

    RANDOM_STATE = 73  # Sheldon's number
    RANDOM_SAMPLES = 100

    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def run() -> None:
        """
        A single entry point to the script, running the entire logic
        :return:
        """
        # make sure the IO is file
        Main.io()
        # baseline graphs
        Main.first_plot()
        # one-dim sensitivity graphs
        Main.second_graph()
        # heatmap sensitivity graphs
        Main.third_graph()

    @staticmethod
    def io() -> None:
        for name in [Main.RESULTS_FOLDER]:
            try:
                os.mkdir(os.path.join(os.path.dirname(__file__), name))
            except:
                pass

    @staticmethod
    def desire_metric(df: dict) -> tuple:
        """
        An approximation to the basic reproduction number
        """
        m = np.asarray(df["y"])
        # TODO: think about it again later
        score = [abs(m[index, 2]/m[index, 3]) for index in range(min([len(m[:, 2]), len(m[:, 3])]))]
        return np.mean(score), np.std(score)

    @staticmethod
    def first_plot() -> None:
        # baseline graph - just run the model and plot it
        initial_conditions = [
            [100, 100, 100, 100],
            [900, 100, 100, 100],
            [100, 900, 100, 100],
            [100, 100, 900, 100],
            [100, 100, 100, 900],
        ]
        for index, initial_condition in enumerate(initial_conditions):
            print("Main.first_plot: baseline for initial condition: {} (#{}/{}-{:.2f}%)".format(initial_condition,
                                                                                                index + 1,
                                                                                                len(initial_conditions),
                                                                                                (index + 1) * 100 / len(
                                                                                                    initial_conditions)))
            Plotter.baseline(model_matrix=Main.solve_the_model(initial_condition=initial_condition),
                             save_path=os.path.join(Main.RESULTS_FOLDER, "baseline_{}.pdf".format(index)))

    @staticmethod
    def second_graph() -> None:
        """
        Generate all the parameter sensitivity graphs
        """
        Main.sens(parameter_range=[-1 + 0.2 * i for i in range(11)], parameter_name="c1")

        Main.sens(parameter_range=[1000 * (i + 1) for i in range(11)], parameter_name="zeta")

        Main.sens(parameter_range=[10 * i for i in range(11)], parameter_name="u")

    @staticmethod
    def sens(parameter_range: list,
             parameter_name: str) -> None:
        """
        This function generates a one-dim sensitivity analysis
        """
        ans_mean = []
        ans_std = []
        for index, parm_val in enumerate(parameter_range):
            print("Main.second_graph: sens for {}={} (#{}/{}-{:.2f}%)".format(parameter_name,
                                                                              parm_val,
                                                                              index + 1,
                                                                              len(parameter_range),
                                                                              (index + 1) * 100 / len(parameter_range)))
            values = [Main.desire_metric(df=Main.solve_the_model_wrapper(params={parameter_name: parm_val},
                                                                         initial_condition=[random.randint(10, 100),
                                                                                            random.randint(1, 100),
                                                                                            random.randint(1, 100)]))[0]
                      for _ in range(Main.RANDOM_SAMPLES)]
            ans_mean.append(np.mean(values))
            ans_std.append(np.std(values))
        Plotter.sensitivity(x=parameter_range,
                            y=ans_mean,
                            y_err=ans_std,
                            x_label=parameter_name,
                            y_label="Average basic reproduction number",
                            save_path=os.path.join(Main.RESULTS_FOLDER, "sensitivity_{}.pdf".format(parameter_name)))

    @staticmethod
    def third_graph() -> None:
        """
        This function responsible to run all the needed heatmap analysis needed for the paper
        """

        Main.heatmap(x=[i * 0.05 for i in range(7)],
                     y=[i * 0.05 for i in range(7)],
                     x_parameter_name="c1",
                     y_parameter_name="c2")

        Main.heatmap(x=[i * 0.1 for i in range(11)],
                     y=[i * 0.1 for i in range(11)],
                     x_parameter_name="gamma1",
                     y_parameter_name="gamma2")

        Main.heatmap(x=[(1+i) * 1000 for i in range(11)],
                     y=[i * 10 for i in range(11)],
                     x_parameter_name="zeta",
                     y_parameter_name="u")

    @staticmethod
    def heatmap(x: list,
                y: list,
                x_parameter_name: str,
                y_parameter_name: str) -> None:
        """
        This function is responsible to get two parameters and return the heatmap of them
        """
        answer = []
        for i_index, x_parm_val in enumerate(x):
            row = []
            for j_index, y_parm_val in enumerate(y):
                print("Main.third_graph: sens for {}={} X {}={} (#{}/{}-{:.2f}%)".format(x_parameter_name,
                                                                                         x_parm_val,
                                                                                         y_parameter_name,
                                                                                         y_parm_val,
                                                                                         i_index * len(
                                                                                             x) + j_index + 1,
                                                                                         len(x) * len(y),
                                                                                         100 * (i_index * len(
                                                                                             x) + j_index + 1) / (
                                                                                                 len(x) * len(
                                                                                             y))))
                values = [Main.desire_metric(df=Main.solve_the_model_wrapper(params={x_parameter_name: x_parm_val,
                                                                                     y_parameter_name: y_parm_val},
                                                                             initial_condition=[random.randint(10, 100),
                                                                                                random.randint(1, 100),
                                                                                                random.randint(1,
                                                                                                               100)]))[
                              0]
                          for _ in range(Main.RANDOM_SAMPLES)]
                row.append(np.mean(values))
            answer.append(row)
        df = pd.DataFrame(data=answer,
                          columns=[round(val, 2) for val in x],
                          index=[round(val, 2) for val in y])
        Plotter.heatmap(df=df,
                        x_label=x_parameter_name,
                        y_label=y_parameter_name,
                        save_path=os.path.join(Main.RESULTS_FOLDER, "heatmap_{}_{}.pdf".format(x_parameter_name,
                                                                                               y_parameter_name)))

    @staticmethod
    def solve_the_model_wrapper(params: dict = None,
                                tspan: list = None,
                                initial_condition: list = None):
        """
        A function responsible to let set the model's parameter values by name
        """
        params = {} if params is None else params
        return Main.solve_the_model(tspan=tspan,
                                    initial_condition=initial_condition,
                                    u=80 if "u" not in params else params["u"],
                                    zeta=10000 if "zeta" not in params else params["zeta"],
                                    c1=0.75 if "c1" not in params else params["c1"],
                                    gamma1=0.5 if "gamma1" not in params else params["gamma1"],
                                    c2=0.75 if "c2" not in params else params["c2"],
                                    gamma2=0.5 if "gamma2" not in params else params["gamma2"])

    @staticmethod
    def solve_the_model(tspan: list = None,
                        initial_condition: list = None,
                        u: float = 80,
                        zeta: float = 10000,
                        c1: float = 0.75,
                        gamma1: float = 0.5,
                        c2: float = 0.75,
                        gamma2: float = 0.5):
        # fix default params
        if tspan is None:
            tspan = [0, 24 * 30]
        if initial_condition is None:
            initial_condition = [1000, 500, 650]

        # make sure the inputs are legit
        if not isinstance(tspan, list) or len(tspan) != 2 or tspan[1] <= tspan[0]:
            raise Exception("Main.solve_the_model: tspan should be a 2-val list [a,b] where b>a.")
        if not isinstance(initial_condition, list) or len(initial_condition) != 4:
            raise Exception("Main.solve_the_model: initial_condition should be a 4-val list.")
        if not (u >= 0 and zeta >= 0 and c2 >= 0 and gamma2 >= 0 and c2 >= 0 and gamma2 >= 0):
            raise Exception("Main.solve_the_model: all parameter values should be non-negative.")

        # load generic script
        with open(os.path.join(os.path.dirname(__file__), Main.OCTAVE_BASED_SCRIPT_NAME), "r") as m_source_file:
            script = m_source_file.read()
        # update the code
        script = script.format(tspan,
                               initial_condition,
                               u,
                               zeta,
                               c1,
                               gamma1,
                               c2,
                               gamma2)
        # save the file for run
        with open(os.path.join(os.path.dirname(__file__), Main.OCTAVE_RUN_SCRIPT_NAME), "w") as m_run_file:
            m_run_file.write(script)
        # run the script
        oct2py.octave.run(os.path.join(os.path.dirname(__file__), Main.OCTAVE_RUN_SCRIPT_NAME))
        # wait to make sure the file is written
        time.sleep(3)
        # load the result file
        return MatFileLoader.read(path=os.path.join(os.path.dirname(__file__), Main.OCTAVE_RUN_RESULT_NAME),
                                  delete_in_end=True)


if __name__ == '__main__':
    Main.run()
