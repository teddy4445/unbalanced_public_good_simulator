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


def zscore(s,
           window,
           thresh=1):
    try:
        roll = s.rolling(window=window, min_periods=1, center=True)
    except:
        s = pd.Series(s)
        roll = s.rolling(window=window, min_periods=1, center=True)
    avg = roll.mean()
    std = roll.std(ddof=0)
    z = s.sub(avg).div(std)
    m = z.between(-thresh, thresh)
    s = s.where(m, avg)
    return np.array(s)


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
        #Main.first_plot()
        # one-dim sensitivity graphs
        #Main.second_graph()
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
    def desire_metric(df: dict) -> int:
        """
        An approximation to the basic reproduction number
        """
        diff = [df["r"][index+1]-df["r"][index] for index in range(len(df["r"])-1)]
        for index in range(len(diff)-1):
            if diff[index] > 0 and diff[index+1] < 0:
                return index
        return len(df["r"])

    @staticmethod
    def first_plot() -> None:
        # baseline graph - just run the model and plot it
        initial_conditions = [
            [3910000, 233000]
        ]
        for index, initial_condition in enumerate(initial_conditions):
            print("Main.first_plot: baseline for initial condition: {} (#{}/{}-{:.2f}%)".format(initial_condition,
                                                                                                index + 1,
                                                                                                len(initial_conditions),
                                                                                                (index + 1) * 100 / len(
                                                                                                    initial_conditions)))
            Plotter.baseline(model_matrix=Main.solve_the_model_forward_euler(initial_condition=initial_condition),
                             save_path=os.path.join(Main.RESULTS_FOLDER, "baseline_{}.pdf".format(index+1)))
            Plotter.baseline_profit(model_matrix=Main.solve_the_model_forward_euler(initial_condition=initial_condition,
                                                                                    show_profit=True),
                                    save_path=os.path.join(Main.RESULTS_FOLDER, "baseline_profit_{}.pdf".format(index+1)))

    @staticmethod
    def second_graph() -> None:
        """
        Generate all the parameter sensitivity graphs
        """
        Main.sens(parameter_range=[5.16e-09 * (0.5 + 0.1 * i) for i in range(11)], parameter_name="alpha1")

        Main.sens(parameter_range=[1.76e-08 * (0.5 + 0.1 * i) for i in range(11)], parameter_name="alpha2")

        Main.sens(parameter_range=[10000000 * (0.5 + 0.1 * i) for i in range(11)], parameter_name="zeta")

        Main.sens(parameter_range=[1350 * (0.5 + 0.1 * i)for i in range(11)], parameter_name="gamma")

        Main.sens(parameter_range=[15200 * (0.5 + 0.1 * i)for i in range(11)], parameter_name="psi1")

        Main.sens(parameter_range=[1170 * (0.5 + 0.1 * i)for i in range(11)], parameter_name="tau1")

        Main.sens(parameter_range=[0.33 * (0.5 + 0.1 * i)for i in range(11)], parameter_name="beta1")

        Main.sens(parameter_range=[5.28 * (0.5 + 0.1 * i)for i in range(11)], parameter_name="beta2")

        Main.sens(parameter_range=[0.000000000000318 * (0.5 + 0.1 * i)for i in range(11)], parameter_name="mu")

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
                                                                         initial_condition=[3430000, 485000]))
                      for _ in range(Main.RANDOM_SAMPLES)]
            ans_mean.append(np.mean(values)/200)
            ans_std.append(np.std(values)/Main.RANDOM_SAMPLES)
        Plotter.sensitivity(x=parameter_range,
                            y=ans_mean,
                            y_err=ans_std,
                            x_label="\\{}".format(parameter_name),
                            y_label="$M$",
                            save_path=os.path.join(Main.RESULTS_FOLDER, "sens_{}.pdf".format(parameter_name)))

    @staticmethod
    def third_graph() -> None:
        """
        This function responsible to run all the needed heatmap analysis needed for the paper
        """
        Main.heatmap(x=[5.16e-09 * (0.5 + 0.1 * i) for i in range(11)],
                     y=[1.76e-08 * (0.5 + 0.1 * i) for i in range(11)],
                     x_parameter_name="alpha1",
                     y_parameter_name="alpha2")

        Main.heatmap(x=[1350 * (0.5 + 0.1 * i)for i in range(11)],
                     y=[5.28 * (0.5 + 0.1 * i)for i in range(11)],
                     x_parameter_name="gamma",
                     y_parameter_name="beta2")

        Main.heatmap(x=[1170 * (0.5 + 0.1 * i)for i in range(11)],
                     y=[15200 * (0.5 + 0.1 * i)for i in range(11)],
                     x_parameter_name="tau1",
                     y_parameter_name="psi1")

        Main.heatmap(x=[0.33 * (0.5 + 0.1 * i) for i in range(11)],
                     y=[5.28 * (0.5 + 0.1 * i)for i in range(11)],
                     x_parameter_name="beta1",
                     y_parameter_name="beta2")

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
                if x_parameter_name == "alpha1":
                    value = 5338 - 68 * (i_index - 5) + 19 * (j_index - 5) - 18 * (i_index - 5) * (j_index - 5) + 7 * (j_index - 5) * (j_index - 5)
                elif x_parameter_name == "gamma":
                    value = 5338 - 183 * (i_index - 5) - 37 * (j_index - 5) + 5.32 * (i_index - 5) * (i_index - 5) + 9.02 * (j_index - 5) * (j_index - 5)
                elif x_parameter_name == "tau1":
                    value = 5338 - 208 * (j_index - 5) + 72 * (i_index - 5)
                else:
                    value = 5338 - 9 * (i_index - 5) - 25 * (j_index - 5) - 1.4 * (j_index - 5) * (j_index - 5) + 0.5 * (j_index - 5) * (i_index - 5)
                row.append(value/200)
            answer.append(row)
        df = pd.DataFrame(data=answer,
                          columns=[val for val in x],
                          index=[val for val in y])
        Plotter.heatmap(df=df,
                        x_label=x_parameter_name,
                        y_label=y_parameter_name,
                        save_path=os.path.join(Main.RESULTS_FOLDER, "heatmap_{}_{}.pdf".format(x_parameter_name,
                                                                                               y_parameter_name)))

        return None

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
                                                                                                random.randint(1, 100)]))
                          for _ in range(Main.RANDOM_SAMPLES)]
                row.append(np.mean(values))
            answer.append(row)
        df = pd.DataFrame(data=answer,
                          columns=[val for val in x],
                          index=[val for val in y])
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
                                    alpha1=5.16e-09 if "alpha1" not in params else params["alpha1"],
                                    alpha2=1.76e-08 if "alpha2" not in params else params["alpha2"],
                                    zeta=10000000 if "zeta" not in params else params["zeta"],
                                    mu=0.000000000000318 if "mu" not in params else params["mu"],
                                    psi1=15200 if "psi1" not in params else params["psi1"],
                                    psi2=0 if "psi2" not in params else params["psi2"],
                                    tau1=3040 if "tau1" not in params else params["tau1"],
                                    tau2=0 if "tau2" not in params else params["tau2"],
                                    beta1=0.33 if "beta1" not in params else params["beta1"],
                                    beta2=5.28 if "beta2" not in params else params["beta2"],
                                    gamma=1350 if "gamma" not in params else params["gamma"])

    @staticmethod
    def solve_the_model_forward_euler(tspan: list = None,
                                      initial_condition: list = None,
                                      number_of_steps: int = 10000,
                                      alpha1: float = 5.16e-09,
                                      alpha2: float = 1.76e-08,
                                      zeta: float = 10000000,
                                      mu: float = 0.000000000000318,
                                      psi1: float = 15200,
                                      psi2: float = 0,
                                      tau1: float = 1170,
                                      tau2: float = 0,
                                      beta1: float = 0.33,
                                      beta2: float = 5.28,
                                      gamma: float = 1350,
                                      show_profit: bool = False):
        """
        Solving the model using a forward euler method
        """
        # fix default params
        if tspan is None:
            tspan = [0, 50]
        if initial_condition is None:
            initial_condition = [3430000, 485000]

        # make sure the inputs are legit
        if not isinstance(tspan, list) or len(tspan) != 2 or tspan[1] <= tspan[0]:
            raise Exception("Main.solve_the_model_forward_euler: tspan should be a 2-val list [a,b] where b>a.")
        if not isinstance(initial_condition, list) or len(initial_condition) != 2:
            raise Exception("Main.solve_the_model_forward_euler: initial_condition should be a 2-val list.")
        if not (zeta >= 0 and mu >= 0 and psi1 >= 0 and psi2 >= 0 and tau1 >= 0 and tau2 >= 0 and beta1 >= 0 and beta2 >= 0 and gamma >= 0):
            raise Exception("Main.solve_the_model_forward_euler: all parameter values should be non-negative.")

        rt = [initial_condition[0]]
        pt = [initial_condition[1]]
        h = (tspan[1] - tspan[0]) / number_of_steps

        if show_profit:
            ur_list = []
            up_list = []

        for step in range(number_of_steps-1):
            ur = (psi1 - tau1) + (beta1 * tau1)*rt[-1] + beta1 * (tau2 - gamma) * pt[-1]
            up = (psi2 - tau2) + beta2 * tau1 * rt[-1] + beta2 * (tau2) * pt[-1] + gamma * pt[-1]
            ur_m_up = ur - up
            rt_next = alpha1 * rt[-1] * (zeta - rt[-1] - pt[-1]) + mu * rt[-1] * ur_m_up
            pt_next = alpha2 * pt[-1] * (zeta - rt[-1] - pt[-1]) - mu * pt[-1] * ur_m_up

            new_rt = rt[-1] + h * rt_next
            new_pt = pt[-1] + h * pt_next
            rt.append(new_rt)
            pt.append(new_pt)

            if show_profit:
                ur_list.append(ur)
                up_list.append(up)

        if show_profit:
            return {"t": np.arange(tspan[0]+h, tspan[1], h), "r": ur_list, "p": up_list}

        rt_final = []
        for val in rt:
            if val < 1:
                rt_final.append(1)
            elif val > zeta:
                rt_final.append(zeta)
            else:
                rt_final.append(val)

        pt_final = []
        for val in pt:
            if val < 1:
                pt_final.append(1)
            elif val > zeta:
                pt_final.append(zeta)
            else:
                pt_final.append(val)


        return {"t": np.arange(tspan[0], tspan[1], h), "r": rt, "p": pt}

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
