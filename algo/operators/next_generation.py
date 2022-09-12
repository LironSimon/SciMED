# library imports
import random

# project imports
from algo.population import Population


class NextGeneration:
    """
    A next generation metric
    """

    def __init__(self):
        pass

    @staticmethod
    def tournament_with_royalty(population: Population,
                                royalty: float):
        """
        A tournament next generation with royalty
        """
        # calc the probability of selecting a gene to a tournament
        sum_fitness = sum(population.get_scores())
        if sum_fitness > 0:
            fitness_probabilities = [score / sum_fitness for score in population.get_scores()]
        else:
            fitness_probabilities = population.get_scores()
        # sort the population by probability of selection
        genes_with_fitness = zip(fitness_probabilities, population.genes)
        genes_with_fitness = sorted(genes_with_fitness, key=lambda x: x[0], reverse=True)
        # pick the most probable genes (those with the largest fitness scores)
        royalty_pop = [val[1] for val in genes_with_fitness[:round(len(genes_with_fitness) * royalty)]]
        # tournament around the other genes
        left_genes = [val[1] for val in genes_with_fitness[round(len(genes_with_fitness) * royalty):]]
        left_fitness = [val[0] for val in genes_with_fitness[round(len(genes_with_fitness) * royalty):]]
        pick_genes = []
        left_count = len(population.genes) - len(royalty_pop)
        while len(pick_genes) < left_count:
            pick_gene = random.choices(left_genes, weights=left_fitness)[0]
            pick_genes.append(pick_gene)
        # add the royalty to the genes selected in the tournament
        pick_genes = list(pick_genes)
        pick_genes.extend(royalty_pop)
        return Population(genes=pick_genes)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<NextGeneration-GA-operator>"
