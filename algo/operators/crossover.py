# library imports
from random import randint

# project imports
from algo.gene import Gene
from algo.population import Population


class Crossover:
    """
    crossover operations class
    """

    def __init__(self):
        pass

    @staticmethod
    def simple(population: Population):
        """
        Just a simple single-point random crossover:
        returns a new population by randomly picking pairs of gene
        parents + a breaking point, to generate a pair of offsprings.
        """
        new_pop = Population()
        while population.size() > 0:
            # pick two different genes
            i = randint(0, population.size()-1)
            j = randint(0, population.size()-1)
            while i == j:
                i = randint(0, population.size()-1)
                j = randint(0, population.size()-1)
            gene_i = population.get(i)
            gene_j = population.get(j)
            # pick a single breaking point
            break_index = randint(1, len(gene_i.feature_indexes)-2)
            # recall to new list
            new_pop.add(gene=Gene(feature_indexes=gene_i.feature_indexes[:break_index]+gene_j.feature_indexes[break_index:]))
            new_pop.add(gene=Gene(feature_indexes=gene_j.feature_indexes[:break_index]+gene_i.feature_indexes[break_index:]))
            # remove from previous list
            # TODO: Teddy, why is the order of removal important? why do we need this if/else
            if i < j:
                population.remove(index=j)
                population.remove(index=i)
            else:
                population.remove(index=i)
                population.remove(index=j)
        return new_pop

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Crossover-GA-operator>"

