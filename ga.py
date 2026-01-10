import numpy as np

class GeneticAlgorithm:
    def __init__(
        self,
        fitness_func,
        chromosome_length,
        pop_size=50,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.05
    ):
        self.fitness_func = fitness_func
        self.chromosome_length = chromosome_length
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.history = []

    def initialize_population(self):
        return np.random.randint(0, 2, (self.pop_size, self.chromosome_length))

    def selection(self, population, fitness):
        idx = np.random.choice(len(population), 2, replace=False)
        return population[idx[np.argmin(fitness[idx])]]

    def crossover(self, p1, p2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.chromosome_length)
            return np.concatenate([p1[:point], p2[point:]])
        return p1.copy()

    def mutate(self, individual):
        for i in range(self.chromosome_length):
            if np.random.rand() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        return individual

    def run(self):
        population = self.initialize_population()

        for _ in range(self.generations):
            fitness = np.array([self.fitness_func(ind) for ind in population])
            self.history.append(np.min(fitness))

            new_population = []
            for _ in range(self.pop_size):
                p1 = self.selection(population, fitness)
                p2 = self.selection(population, fitness)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)

            population = np.array(new_population)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx], self.history
