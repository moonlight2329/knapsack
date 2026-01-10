import numpy as np

class GeneticAlgorithm:
    def __init__(
        self,
        fitness_func,
        bounds,
        pop_size=50,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1
    ):
        self.fitness_func = fitness_func
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        self.dim = len(bounds)
        self.history = []

    def initialize_population(self):
        return np.array([
            [np.random.uniform(low, high) for low, high in self.bounds]
            for _ in range(self.pop_size)
        ])

    def selection(self, population, fitness):
        idx = np.random.choice(len(population), size=2, replace=False)
        return population[idx[np.argmin(fitness[idx])]]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.dim)
            return np.concatenate([parent1[:point], parent2[point:]])
        return parent1.copy()

    def mutate(self, individual):
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                low, high = self.bounds[i]
                individual[i] = np.random.uniform(low, high)
        return individual

    def run(self):
        population = self.initialize_population()

        for gen in range(self.generations):
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
