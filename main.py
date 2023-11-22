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

# help function
def moving_average(a: np.array,
                   n: int = 3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


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
        #Main.second_graph()
        # heatmap sensitivity graphs
        #Main.third_graph()

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
        score = [abs(df["r"][index] / (df["p"][index] + df["r"][index])) for index in range(min([len(df["r"]), len(df["p"])]))]
        return np.mean(score), np.std(score)

    @staticmethod
    def first_plot() -> None:
        # baseline graph - just run the model and plot it
        initial_conditions = [
            [900, 100],
            [100, 900],
            [500, 500],
            [8000, 1000]
        ]
        for index, initial_condition in enumerate(initial_conditions):
            print("Main.first_plot: baseline for initial condition: {} (#{}/{}-{:.2f}%)".format(initial_condition,
                                                                                                index + 1,
                                                                                                len(initial_conditions),
                                                                                                (index + 1) * 100 / len(
                                                                                                    initial_conditions)))
            Plotter.baseline(model_matrix=Main.solve_the_model_forward_euler(initial_condition=initial_condition,
                                                                             gamma=0.4,
                                                                             tau2=0.1,
                                                                             mu=100,
                                                                             ),
                             save_path=os.path.join(Main.RESULTS_FOLDER, "baseline_{}.pdf".format(index)))

    @staticmethod
    def second_graph() -> None:
        """
        Generate all the parameter sensitivity graphs
        """
        Main.sens(parameter_range=[0.2 * i for i in range(11)], parameter_name="beta")

        Main.sens(parameter_range=[1000 * (i + 1) for i in range(11)], parameter_name="zeta")

        Main.sens(parameter_range=[0.02 * i for i in range(11)], parameter_name="gamma")

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
                                                                                            random.randint(1, 100)]))[0]
                      for _ in range(Main.RANDOM_SAMPLES)]
            ans_mean.append(np.mean(values))
            ans_std.append(np.std(values))
        Plotter.sensitivity(x=parameter_range,
                            y=ans_mean,
                            y_err=ans_std,
                            x_label=parameter_name,
                            y_label="$r(t)/(r(t) + p(t))$",
                            save_path=os.path.join(Main.RESULTS_FOLDER, "sensitivity_{}.pdf".format(parameter_name)))

    @staticmethod
    def third_graph() -> None:
        """
        This function responsible to run all the needed heatmap analysis needed for the paper
        """

        Main.heatmap(x=[i * 0.05 for i in range(7)],
                     y=[i * 0.05 for i in range(7)],
                     x_parameter_name="alpha1",
                     y_parameter_name="alpha2")

        Main.heatmap(x=[i * 0.1 for i in range(11)],
                     y=[i * 0.1 for i in range(11)],
                     x_parameter_name="gamma",
                     y_parameter_name="beta")

        Main.heatmap(x=[(1 + i) * 1000 for i in range(11)],
                     y=[i * 10 for i in range(11)],
                     x_parameter_name="zeta",
                     y_parameter_name="mu")

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
                                                                                                random.randint(1, 100)]))[
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
        return Main.solve_the_model_forward_euler(tspan=tspan,
                                    initial_condition=initial_condition,
                                    alpha1=0.000002 if "alpha1" not in params else params["alpha1"],
                                    alpha2=0.000002 if "alpha2" not in params else params["alpha2"],
                                    zeta=10000 if "zeta" not in params else params["zeta"],
                                    mu=0.02 if "mu" not in params else params["mu"],
                                    psi1=1 if "psi1" not in params else params["psi1"],
                                    psi2=1 if "psi2" not in params else params["psi2"],
                                    tau1=0.25 if "tau1" not in params else params["tau1"],
                                    tau2=0.15 if "tau2" not in params else params["tau2"],
                                    beta=1.2 if "beta" not in params else params["beta"],
                                    gamma=0.15 if "gamma" not in params else params["gamma"])

    @staticmethod
    def solve_the_model_forward_euler(tspan: list = None,
                                      initial_condition: list = None,
                                      number_of_steps: int = 1000,
                                      alpha1: float = 0.000002,
                                      alpha2: float = 0.000002,
                                      zeta: float = 10000,
                                      mu: float = 0.02,
                                      psi1: float = 1,
                                      psi2: float = 1,
                                      tau1: float = 0.25,
                                      tau2: float = 0.15,
                                      beta: float = 1.2,
                                      gamma: float = 0.15):
        """
        Solving the model using a forward euler method
        """
        # fix default params
        if tspan is None:
            tspan = [0, 100]
        if initial_condition is None:
            initial_condition = [100, 100]

        # make sure the inputs are legit
        if not isinstance(tspan, list) or len(tspan) != 2 or tspan[1] <= tspan[0]:
            raise Exception("Main.solve_the_model_forward_euler: tspan should be a 2-val list [a,b] where b>a.")
        if not isinstance(initial_condition, list) or len(initial_condition) != 2:
            raise Exception("Main.solve_the_model_forward_euler: initial_condition should be a 2-val list.")
        if not (
                alpha1 >= 0 and alpha2 >= 0 and zeta >= 0 and mu >= 0 and psi1 >= 0 and psi2 >= 0 and tau1 >= 0 and tau2 >= 0 and beta >= 0 and gamma >= 0):
            raise Exception("Main.solve_the_model_forward_euler: all parameter values should be non-negative.")

        rt = [initial_condition[0], initial_condition[0], initial_condition[0]]
        pt = [initial_condition[1], initial_condition[1], initial_condition[1]]
        h = (tspan[1] - tspan[0]) / number_of_steps
        for step in range(number_of_steps-1):
            ur_m_up = ((psi1 - tau1 + beta * tau1) + beta * (tau2 - gamma) * rt[-1]/pt[-1] - (psi2 - tau2 + beta * tau2 + gamma * (1 - beta) - beta * tau2 * pt[-1]))
            rt_next = alpha1 * rt[-1] * (zeta - rt[-1] - pt[-1]) + mu * ur_m_up
            pt_next = alpha2 * pt[-1] * (zeta - pt[-1] - rt[-1]) - mu * ur_m_up

            rt.append(rt[-1] + h * rt_next)
            pt.append(pt[-1] + h * pt_next)

        rt = moving_average(rt, n=9)
        pt = moving_average(pt, n=9)

        return {"t": np.arange(tspan[0]+6*h, tspan[1], h), "r": rt, "p": pt}

    @staticmethod
    def solve_the_model(tspan: list = None,
                        initial_condition: list = None,
                        alpha1: float = 0.01,
                        alpha2: float = 0.01,
                        zeta: float = 10000,
                        mu: float = 0.01,
                        psi1: float = 1,
                        psi2: float = 1,
                        tau1: float = 0.25,
                        tau2: float = 0.15,
                        beta: float = 1.2,
                        gamma: float = 0.15):
        # fix default params
        if tspan is None:
            tspan = [0, 10]
        if initial_condition is None:
            initial_condition = [100, 100]

        # make sure the inputs are legit
        if not isinstance(tspan, list) or len(tspan) != 2 or tspan[1] <= tspan[0]:
            raise Exception("Main.solve_the_model: tspan should be a 2-val list [a,b] where b>a.")
        if not isinstance(initial_condition, list) or len(initial_condition) != 2:
            raise Exception("Main.solve_the_model: initial_condition should be a 2-val list.")
        if not (
                alpha1 >= 0 and alpha2 >= 0 and zeta >= 0 and mu >= 0 and psi1 >= 0 and psi2 >= 0 and tau1 >= 0 and tau2 >= 0 and beta >= 0 and gamma >= 0):
            raise Exception("Main.solve_the_model: all parameter values should be non-negative.")

        # load generic script
        with open(os.path.join(os.path.dirname(__file__), Main.OCTAVE_BASED_SCRIPT_NAME), "r") as m_source_file:
            script = m_source_file.read()
        # update the code
        script = script.format(tspan,
                               initial_condition,
                               alpha1,
                               alpha2,
                               zeta,
                               mu,
                               psi1,
                               psi2,
                               tau1,
                               tau2,
                               beta,
                               gamma)
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
