import config
import random

from core.bird import BirdData, think_bird, update_bird
from typing import List, Any

class Population:
    def __init__(self, size=config.POPULATION_SIZE) -> None:
        self.size = size
        self.generation = 1
        # spawn birds using list comprehension to optimize code
        self.birds = [BirdData() for _ in range(self.size)]

    # being called every 1/60s to check whether alive or not
    def update(self, pipes: List[Any]) -> None:
        for bird_data in self.birds:
            if bird_data.alive:
                think_bird(bird_data, pipes)
                update_bird(bird_data)

    # calculate total alive agents
    def get_alive_count(self) -> int:
        return sum(1 for b in self.birds if b.alive)

    def is_extinct(self) -> bool:
        return self.get_alive_count() == 0

    def calculate_fitness(self) -> None:   
        # square the fitness to increase the prob of being chosen of the best agent      
        transformed_fitness = [b.fitness ** 2 for b in self.birds]
        total_fitness = sum(transformed_fitness)

        if total_fitness == 0:
            for b in self.birds:
                b.fitness = 1
            total_fitness = self.size

            # avoid crash
            b.prob = 1.0 / self.size
            return
        # calculate probability in [0.0, 1.0]
        for b, tf in zip(self.birds, transformed_fitness):
            b.prob = tf / total_fitness
    
    # roulette wheel selection
    def select_parent(self) -> BirdData:
        r = random.random()
        index = 0
        while r > 0 and index < len(self.birds):
            r -= self.birds[index].prob
            index += 1
        index -= 1
        return self.birds[index]

    def next_generation(self) -> None:
        self.calculate_fitness()
        new_birds = []

        # sorting fitness score in a descending way
        sorted_birds = sorted(self.birds, key=lambda b: b.fitness, reverse=True)

        # clone 10 best birds (elitism)
        for i in range(config.NUM_ELITISM):
            elite_brain = sorted_birds[i].brain.copy()

            # spawn new data to inherit traits of that brain
            elite_clone = BirdData(brain=elite_brain)
            new_birds.append(elite_clone)

        # crossover the remaining, except the 10 best birds
        for _ in range(self.size - config.NUM_ELITISM):
            parent_A = self.select_parent()
            parent_B = self.select_parent()

            child_brain = parent_A.brain.crossover(parent_B.brain)
            child_brain.mutate(config.MUTATION_RATE)
            
            # spawn new data with children's brain
            child_clone = BirdData(brain=child_brain)
            new_birds.append(child_clone)

        # delete old gen, spawn new gen
        self.birds = new_birds
        self.generation += 1