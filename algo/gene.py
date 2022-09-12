# library imports
from random import randint
import pandas as pd

# project imports


class Gene:
    """
    A gene to represent a subset of features,
    ensuring there is only one feature representation
    from each group of similarly-created feature.
    """

    def __init__(self,
                 feature_indexes: list,
                 scores: pd.DataFrame = pd.DataFrame(),
                 fitness: float = 0,
                 model_object=None):
        self.feature_indexes = feature_indexes
        self.scoring_history = scores
        self.fitness = fitness
        self.model_object = model_object


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Gene | feature_indexes = {}, fitness = {}>".format(self.feature_indexes,
                                                                    self.fitness)

    @staticmethod
    def random(feature_indexes_ranges: list,
               feature_count: int):
        return Gene(feature_indexes=[randint(feature_indexes_ranges[i][0],
                                             feature_indexes_ranges[i][1])
                                     for i in range(feature_count)])

    def length(self):
        return len(self.feature_indexes)

