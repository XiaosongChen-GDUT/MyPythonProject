# Assuming these are imported from your existing code
from Algorithm.PathPlanning import Path_Planning
from Program.DataModel import Model
from Algorithm.slap import slap
from Program.DataModel import Model
import random
import numpy as np
import matplotlib.pyplot as plt
import geatpy as ea
from collections import defaultdict
row_space = 1.54
h1 = 3.15
h2 = 2.55
h3 = 1.5
heights = [h1, h2, h3]
Acc_car = 0.3;
Dec_car = 0.3;
Max_speed_car = 1.2;
Switching_time = 4;
Acc_lift = 0.15;
Dec_lift = 0.15;
Max_speed_lift = 1.4;
out_point = [445, 820, 971, 1156]
enter_node = [51, 348, 636, 925, 1110, 1620]
fist_connect_point = [642,  674, 1116, 1148]
second_connect_point = [2374,  2406, 2844, 2876]
third_connect_point = [3899, 4135]


# Constants from your code (for reference)
row_space = 1.54  # 排间距
heights = [3.15, 2.55, 1.5]  # 各楼层高度
enter_node = [51, 348, 636, 925, 1110, 1620]  # 入口点
out_point = [445, 820, 971, 1156]  # 出口点
fist_connect_point = [642, 674, 1116, 1148]  # 1楼接驳点
second_connect_point = [2374, 2406, 2844, 2876]  # 2楼接驳点
third_connect_point = [3899, 4135]  # 3楼接驳点

# 1. Chromosome Class
class Chromosome:
    """Represents an individual's decision variables (chromosome) as a sequence of aisle IDs."""
    def __init__(self, length, aisle_ids):
        """
        Initialize a chromosome with random aisle IDs.

        :param length: Number of decision variables (e.g., number of free aisles).
        :param aisle_ids: List of available aisle IDs to choose from.
        """
        self.length = length
        self.aisle_ids = aisle_ids
        # Randomly select 'length' aisle IDs from available ones
        self.genes = [random.choice(self.aisle_ids) for _ in range(self.length)]

    def __repr__(self):
        """String representation of the chromosome for debugging."""
        return f"Chromosome({self.genes})"

# 2. Individual Class
class Individual:
    """Represents a single solution in the population, containing a chromosome and objectives."""
    def __init__(self, chromosome):
        """
        Initialize an individual with a chromosome.

        :param chromosome: Chromosome object representing decision variables.
        """
        self.chromosome = chromosome
        self.objectives = None  # Will be set after evaluation (F1, F2, F3, F4)
        self.cv = 0.0  # Constraint violation value, initialized to 0

    def evaluate(self, problem):
        """
        Evaluate the individual using the problem's objective function.

        :param problem: LocationAssignment_Aisles instance to compute objectives.
        """
        # Convert chromosome genes to phenotype for evaluation
        phen = np.array([self.chromosome.genes])
        pop = ea.Population('RI', problem.Field, 1)  # Create a temporary population
        pop.Phen = phen
        problem.aimFunc(pop)  # Call the problem's objective function
        self.objectives = pop.ObjV[0]  # Normalized objective values
        self.cv = pop.CV[0, 0] if pop.CV is not None else 0.0

# 3. Population Class
class Population:
    """Manages a collection of individuals."""
    def __init__(self, size, problem):
        """
        Initialize a population with random individuals.

        :param size: Number of individuals in the population.
        :param problem: LocationAssignment_Aisles instance for evaluation.
        """
        self.size = size
        self.problem = problem
        # Generate individuals with chromosomes based on free aisles
        self.individuals = [
            Individual(Chromosome(len(problem.free_aisles), problem.free_aisles))
            for _ in range(size)
        ]
        self.evaluate()

    def evaluate(self):
        """Evaluate all individuals in the population."""
        for ind in self.individuals:
            ind.evaluate(self.problem)

# 4. Evolution Class
class Evolution:
    """Handles the NSGA-II evolutionary process."""
    def __init__(self, problem, population_size, max_generations):
        """
        Initialize the evolution process with NSGA-II settings.

        :param problem: LocationAssignment_Aisles instance.
        :param population_size: Size of the population.
        :param max_generations: Number of generations to evolve.
        """
        self.problem = problem
        self.population_size = population_size
        self.max_generations = max_generations
        self.algorithm = None
        self.setup_algorithm()

    def setup_algorithm(self):
        """Configure the NSGA-II algorithm using geatpy."""
        # Create a geatpy Population object for algorithm compatibility
        Encoding = 'RI'  # Real Integer encoding
        Field = ea.crtfld(Encoding, self.problem.varTypes, self.problem.ranges, self.problem.borders)
        pop = ea.Population(Encoding, Field, self.population_size)
        #Heuristic initialization: sort aisles by capacity
        sorted_aisles = sorted(self.problem.asiles.keys(), key=lambda x: self.problem.asiles[x]['capacity'], reverse=True)
        initial_phen = np.array([sorted_aisles[:self.problem.Dim] for _ in range(self.population_size)])
        # Set the initial phenotype (and chromosome for 'RI' encoding)
        pop.Phen = initial_phen
        pop.Chrom = initial_phen  # Ensure consistency; for 'RI', Phen == Chrom
        # Initialize algorithm template
        self.algorithm = ea.moea_NSGA2_templet(self.problem, pop)
        self.algorithm.MAXGEN = self.max_generations
        self.algorithm.mutOper.CR = 0.2  # Mutation probability
        self.algorithm.recOper.XOVR = 0.9  # Crossover probability
        self.algorithm.logTras = 1  # Log every generation
        self.algorithm.verbose = True  # Print logs
        self.algorithm.drawing = 1  # Draw Pareto front
        self.algorithm.paretoFront = np.array([#设置参考帕累托前沿面
            [0.5, 10.0, 100.0, 50.0],  # 一个理想点
            [1.0, 20.0, 80.0, 40.0],   # 另一个理想点
            [0.8, 15.0, 120.0, 60.0],
            [0.6, 12.0, 140.0, 70.0],
            # 根据实际情况添加更多点
        ])  # 使用最终解集作为参考前沿

    def run(self):
        """Execute the NSGA-II evolution and return results."""
        # Initialize population with heuristic (e.g., sorted by capacity)
        # Run the algorithm
        NDSet, final_pop = self.algorithm.run()
        return NDSet, final_pop

# 5. Plotter Class
class Plotter:
    """Tools for visualizing the Pareto front and evolution metrics."""
    @staticmethod
    def plot_pareto_front(ndset):
        """
        Plot the Pareto front (first two objectives for simplicity).

        :param ndset: Non-dominated set (geatpy Population object).
        """
        if ndset.sizes == 0:
            print("No feasible solutions to plot!")
            return
        objv = ndset.ObjV  # Objective values
        plt.scatter(objv[:, 0], objv[:, 1], c='blue', label='Pareto Front')
        plt.xlabel('Normalized Objective 1 (Gravity Center)')
        plt.ylabel('Normalized Objective 2 (Efficiency)')
        plt.title('Pareto Front (Objectives 1 vs 2)')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_metrics(log):
        """
        Plot evolution metrics like IGD and HV.

        :param log: Algorithm log dictionary from geatpy.
        """
        if log is None:
            return
        metrics = np.array([log['igd'], log['hv']]).T
        labels = [['IGD'], ['HV']]
        ea.trcplot(metrics, labels, titles=[['IGD'], ['HV']])

# 6. DataSaver Class
class DataSaver:
    """Saves algorithm results to a file."""
    @staticmethod
    def save_results(ndset, filename='pareto_results.csv'):
        """
        Save Pareto front solutions to a CSV file.

        :param ndset: Non-dominated set (geatpy Population object).
        :param filename: Output file name.
        """
        if ndset.sizes == 0:
            print("No feasible solutions to save!")
            return
        data = np.hstack((ndset.Phen, ndset.ObjV))
        headers = ['Aisle_' + str(i) for i in range(ndset.Phen.shape[1])] + \
                  ['Obj1_Gravity', 'Obj2_Efficiency', 'Obj3_Balance', 'Obj4_Relatedness']
        np.savetxt(filename, data, delimiter=',', header=','.join(headers), comments='')
        print(f"Results saved to {filename}")

# 7. Runner Class
class Runner:
    """Coordinates the execution of the NSGA-II algorithm."""
    def __init__(self, problem, population_size=50, max_generations=100):
        """
        Initialize the runner with problem and algorithm parameters.

        :param problem: LocationAssignment_Aisles instance.
        :param population_size: Number of individuals.
        :param max_generations: Number of generations.
        """
        self.problem = problem
        self.population_size = population_size
        self.max_generations = max_generations
        self.evolution = Evolution(problem, population_size, max_generations)

    def run(self):
        """Run the NSGA-II algorithm and process results."""
        NDSet, final_pop = self.evolution.run()
        Plotter.plot_pareto_front(NDSet)
        Plotter.plot_metrics(self.evolution.algorithm.log)
        DataSaver.save_results(NDSet)
        return NDSet, final_pop

# Main Function
if __name__ == '__main__':
    # Initialize problem instance
    model = Model()
    path_planning = Path_Planning(model)
    problem = slap()
    problem.set_model(model)
    problem.set_path_planning(path_planning)

    # Generate test data
    print("Correlation Matrix:\n", problem.correlation)
    print("Pending Items:\n", problem.test_items)

    # # Initialize the problem
    # problem.initProblem()

    # Run NSGA-II
    runner = Runner(problem, population_size=50, max_generations=100)
    NDSet, final_pop = runner.run()

    # Output results
    print(f"Execution Time: {runner.evolution.algorithm.passTime} seconds")
    print(f"Number of Non-dominated Solutions: {NDSet.sizes}")
    if NDSet.sizes > 0 and runner.evolution.algorithm.log:
        print(f"Final GD: {runner.evolution.algorithm.log['gd'][-1]}")
        print(f"Final IGD: {runner.evolution.algorithm.log['igd'][-1]}")
        print(f"Final HV: {runner.evolution.algorithm.log['hv'][-1]}")
        print(f"Final Spacing: {runner.evolution.algorithm.log['spacing'][-1]}")
