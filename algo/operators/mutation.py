# library imports
import numpy as np
from random import randint, random

# project imports
from algo.population import Population


class Mutation:
    """
    mutation operations class
    """

    def __init__(self):
        pass

    @staticmethod
    def simple(population: Population,
               feature_indexes_ranges: list,
               mutation_rate: float,
               w: list = None):
        """
        Just a simple random mutation: changes the selected  feature
        of one feature-group to a different feature from the same group.
        """
        if w is None:
            w = [1 / len(population[0].feature_indexes) for _ in range(len(population[0].feature_indexes))]
        w = np.asarray(w)
        w = w/w.sum()
        for gene in population:
            if random() < mutation_rate:
                pick_index = np.random.choice(range(len(gene.feature_indexes)),
                                              1,
                                              p=w)[0]
                gene.feature_indexes[pick_index] = randint(feature_indexes_ranges[pick_index][0],
                                                           feature_indexes_ranges[pick_index][1])
        return population

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Mutation-GA-operator>"

