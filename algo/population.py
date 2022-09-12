# library imports

# project imports
from algo.gene import Gene


class Population:
    """
    A population of genes
    """

    def __init__(self,
                 genes: list = None):
        self.genes = genes if isinstance(genes, list) and len(genes) > 0 else []

    def __getitem__(self, item):
        return self.genes[item]  # delegate to li.__getitem__

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Population | size = {}>".format(len(self.genes))

    @staticmethod
    def random(size: int,
               feature_count: int,
               feature_indexes_ranges: list):
        return Population(genes=[Gene.random(feature_indexes_ranges=feature_indexes_ranges,
                                             feature_count=feature_count)
                                 for _ in range(size)])

    def get_best(self):
        best_gene = self.genes[0]
        best_gene_fitness = self.genes[0].fitness
        for gene in self.genes:
            if gene.fitness > best_gene_fitness:
                best_gene_fitness = gene.fitness
                best_gene = gene
        return best_gene

    def get_scores(self):
        return [gene.fitness for gene in self.genes]

    def get(self,
            index: int):
        return self.genes[index]

    def remove(self,
               index: int):
        self.genes.remove(self.genes[index])

    def add(self,
            gene: Gene):
        self.genes.append(gene)

    def size(self):
        return len(self.genes)
