import pygad
import numpy
import time
import matplotlib.pyplot as plt

S25 = [1, 2, 3, 5, 10, 17, 25, 29, 30, 41, 51, 60, 73, 79, 80, 82, 86, 90, 94, 100, 106, 108, 110, 120, 138]
S35 = [1, 2, 3, 5, 10, 17, 25, 29, 30, 41, 51, 60, 73, 79, 80, 82, 86, 90, 94, 100, 106, 108, 110, 120, 140, 131, 133,
       138, 142, 161, 180, 190, 200, 201, 207]
S45 = [1, 2, 3, 5, 10, 17, 27, 29, 30, 41, 51, 60, 73, 79, 80, 82, 86, 90, 94, 100, 106, 108, 110, 120, 140, 131, 133,
       138, 142, 161, 180, 190, 200, 201, 202, 203, 204, 205, 210, 220, 230, 240, 242, 243, 250, ]

measuredTimes = []
quantityOfElments = [len(S25), len(S35), len(S45)]

gene_space = [0, 1]
sol_per_pop = 10
num_parents_mating = 6
num_generations = 100
keep_parents = 3
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 5
stop_criteria = "reach_0.0"
appLoopsQuantity = 10
summedTime = 0


def fitness_func25(solution, solution_idx):
    sum1 = numpy.sum(solution * S25)
    solution_invert = 1 - solution
    sum2 = numpy.sum(solution_invert * S25)
    fitness = -numpy.abs(sum1 - sum2)
    return fitness


fitness_function25 = fitness_func25
num_genes = len(S25)

for i in range(appLoopsQuantity):
    start = time.time()

    ga_instance25 = pygad.GA(gene_space=gene_space,
                             num_generations=num_generations,
                             num_parents_mating=num_parents_mating,
                             fitness_func=fitness_function25,
                             sol_per_pop=sol_per_pop,
                             num_genes=num_genes,
                             parent_selection_type=parent_selection_type,
                             keep_parents=keep_parents,
                             crossover_type=crossover_type,
                             mutation_type=mutation_type,
                             mutation_percent_genes=mutation_percent_genes,
                             stop_criteria=stop_criteria)

    ga_instance25.run()

    end = time.time()
    elapsed_time = (end - start)
    summedTime += elapsed_time

measuredTimes.append(summedTime / appLoopsQuantity)

print("Results for 25 elements: ")
solution, solution_fitness, solution_idx = ga_instance25.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

ga_instance25.plot_fitness()


def fitness_func35(solution, solution_idx):
    sum1 = numpy.sum(solution * S35)
    solution_invert = 1 - solution
    sum2 = numpy.sum(solution_invert * S35)
    fitness = -numpy.abs(sum1 - sum2)
    return fitness


fitness_function35 = fitness_func35
num_genes = len(S35)

summedTime = 0
for i in range(appLoopsQuantity):
    start = time.time()

    ga_instance35 = pygad.GA(gene_space=gene_space,
                             num_generations=num_generations,
                             num_parents_mating=num_parents_mating,
                             fitness_func=fitness_function35,
                             sol_per_pop=sol_per_pop,
                             num_genes=num_genes,
                             parent_selection_type=parent_selection_type,
                             keep_parents=keep_parents,
                             crossover_type=crossover_type,
                             mutation_type=mutation_type,
                             mutation_percent_genes=mutation_percent_genes,
                             stop_criteria=stop_criteria)

    ga_instance35.run()

    end = time.time()
    elapsed_time = (end - start)
    summedTime += elapsed_time

measuredTimes.append(summedTime / appLoopsQuantity)
print("Results for 35 elements: ")
solution, solution_fitness, solution_idx = ga_instance35.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

ga_instance35.plot_fitness()


def fitness_func45(solution, solution_idx):
    sum1 = numpy.sum(solution * S45)
    solution_invert = 1 - solution
    sum2 = numpy.sum(solution_invert * S45)
    fitness = -numpy.abs(sum1 - sum2)
    return fitness


fitness_function45 = fitness_func45

num_genes = len(S45)

summedTime = 0
for i in range(appLoopsQuantity):
    start = time.time()

    ga_instance45 = pygad.GA(gene_space=gene_space,
                             num_generations=num_generations,
                             num_parents_mating=num_parents_mating,
                             fitness_func=fitness_function45,
                             sol_per_pop=sol_per_pop,
                             num_genes=num_genes,
                             parent_selection_type=parent_selection_type,
                             keep_parents=keep_parents,
                             crossover_type=crossover_type,
                             mutation_type=mutation_type,
                             mutation_percent_genes=mutation_percent_genes,
                             stop_criteria=stop_criteria)

    ga_instance45.run()

    end = time.time()
    elapsed_time = (end - start)
    summedTime += elapsed_time

measuredTimes.append(summedTime / appLoopsQuantity)

print("Results for 45 elements: ")

solution, solution_fitness, solution_idx = ga_instance45.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

ga_instance45.plot_fitness()

print("AVG time: " + str(measuredTimes))
plt.plot(quantityOfElments, measuredTimes, color='r', marker='o')
plt.ylabel('Srednia pomierzonych czasow')
plt.xlabel('Liczba elementow')
plt.title('Czas dzialania programu w zaleznosci od il. elementow')
plt.show()
