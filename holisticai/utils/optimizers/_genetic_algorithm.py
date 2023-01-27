import sys
import time

import numpy as np
from tqdm import tqdm


class GAHiperparameters(object):
    def __init__(self, **kargs):
        self.max_num_iteration = kargs.get("max_num_iteration", None)
        self.population_size = kargs.get("population_size", 100)
        self.mutation_probability = kargs.get("mutation_probability", 0.1)
        self.elit_ratio = kargs.get("elit_ratio", 0.01)
        self.crossover_probability = kargs.get("crossover_probability", 0.5)
        self.parents_portion = kargs.get("parents_portion", 0.3)
        self.crossover_type = kargs.get("crossover_type", "uniform")
        self.max_iteration_without_improv = kargs.get(
            "max_iteration_without_improv", None
        )


class GeneticAlgorithm(object):
    """
    Genetic Algorithm (Elitist version)
    Implementation of elitist genetic algorithm for solving problems with
    continuous, integers, or mixed variables.
    """

    def __init__(
        self,
        function,
        dimension: int,
        variable_type: str = "bool",
        variable_boundaries: np.ndarray = None,
        variable_type_mixed: np.ndarray = None,
        algorithm_parameters: GAHiperparameters = None,
        verbose: int = 0,
    ):
        """
        Parameters
        ----------

        function : <Callable>
        The given objective function to be minimized.

        dimension : int
        The number of decision variables.

        variable_type: str
            - 'bool' if all variables are Boolean;
            - 'int' if all variables are integer; and
            - 'real' if all variables are real value or continuous

        for mixed type see @param variable_type_mixed.

        variable_boundaries: array-like
        None if variable_type is 'bool'; otherwise provide an array of tuples
        of length two as boundaries for each variable;
        the length of the array must be equal dimension. For example,
        np.array([0,100],[0,200]) determines lower boundary 0 and upper boundary 100 for first
        and upper boundary 200 for second variable where dimension is 2.

        variable_type_mixed: array-like
        None if all variables have the same type; otherwise this can be used to
        specify the type of each variable separately. For example if the first
        variable is integer but the second one is real the input is:
        np.array(['int'],['real']). NOTE: it does not accept 'bool'. If variable
        type is Boolean use 'int' and provide a boundary as [0,1]
        in variable_boundaries. Also if variable_type_mixed is applied,
        variable_boundaries has to be defined.

        algorithm_parameters: GAArguments
        class containing GA algoirhtm hiperparameters

        verbose: int
        If >0, log information

        """
        self.verbose = verbose

        # input function
        assert callable(function), "function must be callable"
        self.f = function
        self.dim = int(dimension)

        # input variable type
        assert (
            variable_type == "bool" or variable_type == "int" or variable_type == "real"
        ), "\n variable_type must be 'bool', 'int', or 'real'"

        # input variables' type (MIXED)
        if variable_type_mixed is None:

            if variable_type == "real":
                self.var_type = np.array([["real"]] * self.dim)
            else:
                self.var_type = np.array([["int"]] * self.dim)
        else:
            assert (
                type(variable_type_mixed).__module__ == "numpy"
            ), "\n variable_type must be numpy array"
            assert (
                len(variable_type_mixed) == self.dim
            ), "\n variable_type must have a length equal dimension."

            for i in variable_type_mixed:
                assert i == "real" or i == "int", (
                    "\n variable_type_mixed is either 'int' or 'real' "
                    + "ex:['int','real','real']"
                    + "\n for 'boolean' use 'int' and specify boundary as [0,1]"
                )

            self.var_type = variable_type_mixed

        # input variables' boundaries
        if variable_type != "bool" or type(variable_type_mixed).__module__ == "numpy":

            assert (
                type(variable_boundaries).__module__ == "numpy"
            ), "\n variable_boundaries must be numpy array"

            assert (
                len(variable_boundaries) == self.dim
            ), "\n variable_boundaries must have a length equal dimension"

            for i in variable_boundaries:
                assert (
                    len(i) == 2
                ), "\n boundary for each variable must be a tuple of length two."
                assert (
                    i[0] <= i[1]
                ), "\n lower_boundaries must be smaller than upper_boundaries [lower,upper]"
            self.var_bound = variable_boundaries
        else:
            self.var_bound = np.array([[0, 1]] * self.dim)

        # input algorithm's parameters
        self.param = algorithm_parameters.__dict__

        self.pop_s = int(self.param["population_size"])

        assert (
            self.param["parents_portion"] <= 1 and self.param["parents_portion"] >= 0
        ), "parents_portion must be in range [0,1]"

        self.par_s = int(self.param["parents_portion"] * self.pop_s)
        trl = self.pop_s - self.par_s
        if trl % 2 != 0:
            self.par_s += 1

        self.prob_mut = self.param["mutation_probability"]

        assert (
            self.prob_mut <= 1 and self.prob_mut >= 0
        ), "mutation_probability must be in range [0,1]"

        self.prob_cross = self.param["crossover_probability"]
        assert (
            self.prob_cross <= 1 and self.prob_cross >= 0
        ), "mutation_probability must be in range [0,1]"

        assert (
            self.param["elit_ratio"] <= 1 and self.param["elit_ratio"] >= 0
        ), "elit_ratio must be in range [0,1]"

        trl = self.pop_s * self.param["elit_ratio"]
        if trl < 1 and self.param["elit_ratio"] > 0:
            self.num_elit = 1
        else:
            self.num_elit = int(trl)

        assert (
            self.par_s >= self.num_elit
        ), "\n number of parents must be greater than number of elits"

        if self.param["max_num_iteration"] == None:
            self.iterate = 0
            for i in range(0, self.dim):
                if self.var_type[i] == "int":
                    self.iterate += (
                        (self.var_bound[i][1] - self.var_bound[i][0])
                        * self.dim
                        * (100 / self.pop_s)
                    )
                else:
                    self.iterate += (
                        (self.var_bound[i][1] - self.var_bound[i][0])
                        * 50
                        * (100 / self.pop_s)
                    )
            self.iterate = int(self.iterate)
            if (self.iterate * self.pop_s) > 10000000:
                self.iterate = 10000000 / self.pop_s
        else:
            self.iterate = int(self.param["max_num_iteration"])

        self.c_type = self.param["crossover_type"]
        assert (
            self.c_type == "uniform"
            or self.c_type == "one_point"
            or self.c_type == "two_point"
        ), "\n crossover_type must 'uniform', 'one_point', or 'two_point' Enter string"

        self.stop_mniwi = False
        if self.param["max_iteration_without_improv"] == None:
            self.mniwi = self.iterate + 1
        else:
            self.mniwi = int(self.param["max_iteration_without_improv"])

    def run(self):

        # Initial Population
        self.integers = np.where(self.var_type == "int")
        self.reals = np.where(self.var_type == "real")

        pop = np.array([np.zeros(self.dim + 1)] * self.pop_s)
        solo = np.zeros(self.dim + 1)
        var = np.zeros(self.dim)

        for p in range(0, self.pop_s):

            for i in self.integers[0]:
                var[i] = np.random.randint(
                    self.var_bound[i][0], self.var_bound[i][1] + 1
                )
                solo[i] = var[i].copy()
            for i in self.reals[0]:
                var[i] = self.var_bound[i][0] + np.random.random() * (
                    self.var_bound[i][1] - self.var_bound[i][0]
                )
                solo[i] = var[i].copy()

            obj = self.sim(var)
            solo[self.dim] = obj
            pop[p] = solo.copy()

        # Report
        self.report = []
        self.test_obj = obj
        self.best_variable = var.copy()
        self.best_function = obj

        t = 1
        counter = 0
        with tqdm(total=self.iterate) as pbar:
            while t <= self.iterate:

                if self.verbose == True:
                    pbar.update(1)

                # Sort
                pop = pop[pop[:, self.dim].argsort()]

                if pop[0, self.dim] < self.best_function:
                    counter = 0
                    self.best_function = pop[0, self.dim].copy()
                    self.best_variable = pop[0, : self.dim].copy()
                    pbar.set_description(f"Cost: {self.best_function:.4f}")
                    pbar.refresh()
                else:
                    counter += 1
                # Report
                self.report.append(pop[0, self.dim])

                # Normalizing objective function
                normobj = np.zeros(self.pop_s)

                minobj = pop[0, self.dim]
                if minobj < 0:
                    normobj = pop[:, self.dim] + abs(minobj)

                else:
                    normobj = pop[:, self.dim].copy()

                maxnorm = np.amax(normobj)
                normobj = maxnorm - normobj + 1

                # Calculate probability
                sum_normobj = np.sum(normobj)
                prob = np.zeros(self.pop_s)
                prob = normobj / sum_normobj
                cumprob = np.cumsum(prob)

                # Select parents
                par = np.array([np.zeros(self.dim + 1)] * self.par_s)

                for k in range(0, self.num_elit):
                    par[k] = pop[k].copy()
                for k in range(self.num_elit, self.par_s):
                    index = np.searchsorted(cumprob, np.random.random())
                    par[k] = pop[index].copy()

                ef_par_list = np.array([False] * self.par_s)
                par_count = 0
                while par_count == 0:
                    for k in range(0, self.par_s):
                        if np.random.random() <= self.prob_cross:
                            ef_par_list[k] = True
                            par_count += 1

                ef_par = par[ef_par_list].copy()

                # New generation
                pop = np.array([np.zeros(self.dim + 1)] * self.pop_s)

                for k in range(0, self.par_s):
                    pop[k] = par[k].copy()

                for k in range(self.par_s, self.pop_s, 2):
                    r1 = np.random.randint(0, par_count)
                    r2 = np.random.randint(0, par_count)
                    pvar1 = ef_par[r1, : self.dim].copy()
                    pvar2 = ef_par[r2, : self.dim].copy()

                    ch = self.cross(pvar1, pvar2, self.c_type)
                    ch1 = ch[0].copy()
                    ch2 = ch[1].copy()

                    ch1 = self.mut(ch1)
                    ch2 = self.mutmidle(ch2, pvar1, pvar2)
                    solo[: self.dim] = ch1.copy()
                    obj = self.sim(ch1)
                    solo[self.dim] = obj
                    pop[k] = solo.copy()
                    solo[: self.dim] = ch2.copy()
                    obj = self.sim(ch2)
                    solo[self.dim] = obj
                    pop[k + 1] = solo.copy()

                t += 1
                if counter > self.mniwi:
                    pop = pop[pop[:, self.dim].argsort()]
                    if pop[0, self.dim] >= self.best_function:
                        t = self.iterate
                        if self.verbose == True:
                            pbar.update(t - pbar.n)
                        time.sleep(2)
                        t += 1
                        self.stop_mniwi = True

        # Sort
        pop = pop[pop[:, self.dim].argsort()]

        if pop[0, self.dim] < self.best_function:

            self.best_function = pop[0, self.dim].copy()
            self.best_variable = pop[0, : self.dim].copy()
            pbar.set_description(f"Cost: {self.best_function:.4f}")
            pbar.refresh()

        # Report
        self.report.append(pop[0, self.dim])

        self.output_dict = {
            "variable": self.best_variable,
            "function": self.best_function,
        }
        if self.verbose == True:
            show = " " * 100
            sys.stdout.write("\r%s" % (show))
            sys.stdout.write("\r The best solution found:\n %s" % (self.best_variable))
            sys.stdout.write("\n\n Objective function:\n %s\n" % (self.best_function))
            sys.stdout.flush()
        re = np.array(self.report)

        if self.stop_mniwi == True:
            sys.stdout.write(
                "\nWarning: GA is terminated due to the"
                + " maximum number of iterations without improvement was met!"
            )

    def cross(self, x, y, c_type):

        ofs1 = x.copy()
        ofs2 = y.copy()

        if c_type == "one_point":
            ran = np.random.randint(0, self.dim)
            for i in range(0, ran):
                ofs1[i] = y[i].copy()
                ofs2[i] = x[i].copy()

        if c_type == "two_point":

            ran1 = np.random.randint(0, self.dim)
            ran2 = np.random.randint(ran1, self.dim)

            for i in range(ran1, ran2):
                ofs1[i] = y[i].copy()
                ofs2[i] = x[i].copy()

        if c_type == "uniform":

            for i in range(0, self.dim):
                ran = np.random.random()
                if ran < 0.5:
                    ofs1[i] = y[i].copy()
                    ofs2[i] = x[i].copy()

        return np.array([ofs1, ofs2])

    def mut(self, x):

        for i in self.integers[0]:
            ran = np.random.random()
            if ran < self.prob_mut:

                x[i] = np.random.randint(self.var_bound[i][0], self.var_bound[i][1] + 1)

        for i in self.reals[0]:
            ran = np.random.random()
            if ran < self.prob_mut:

                x[i] = self.var_bound[i][0] + np.random.random() * (
                    self.var_bound[i][1] - self.var_bound[i][0]
                )

        return x

    def mutmidle(self, x, p1, p2):
        for i in self.integers[0]:
            ran = np.random.random()
            if ran < self.prob_mut:
                if p1[i] < p2[i]:
                    x[i] = np.random.randint(p1[i], p2[i])
                elif p1[i] > p2[i]:
                    x[i] = np.random.randint(p2[i], p1[i])
                else:
                    x[i] = np.random.randint(
                        self.var_bound[i][0], self.var_bound[i][1] + 1
                    )

        for i in self.reals[0]:
            ran = np.random.random()
            if ran < self.prob_mut:
                if p1[i] < p2[i]:
                    x[i] = p1[i] + np.random.random() * (p2[i] - p1[i])
                elif p1[i] > p2[i]:
                    x[i] = p2[i] + np.random.random() * (p1[i] - p2[i])
                else:
                    x[i] = self.var_bound[i][0] + np.random.random() * (
                        self.var_bound[i][1] - self.var_bound[i][0]
                    )
        return x

    def evaluate(self):
        return self.f(self.temp)

    def sim(self, X):
        self.temp = X.copy()
        obj = self.evaluate()
        return obj
