# library imports
import numpy as np
import pandas as pd

# project imports
from algo.ebs.eq_node import EqNode


class Eq:
    """
    This class represents an equation that constructed from simple functions
    """

    def __init__(self,
                 tree: EqNode):
        self.tree = tree
        self.linear_a = 1
        self.linear_b = 0

    def eval(self,
             x_values: pd.DataFrame) -> pd.Series:
        """
        Eval the y_pred from the input
        :param x_values: the input
        """
        return self.linear_a * self.tree.eval(x_values=x_values) + self.linear_b

    def predict(self,
                x_values: pd.DataFrame) -> pd.Series:
        """
        Eval the y_pred from the input
        :param x_values: the input
        """
        return self.linear_a * self.tree.eval(x_values=x_values) + self.linear_b

    def fix_nodes(self) -> None:
        """ fix nodes' is_leaf flag if has been corrupted by other process """
        return self.tree.fix_node()

    def to_id_str(self) -> str:
        """ print equation in a narrow way for hash mapping """
        return self.tree.to_id_str()

    def to_string(self) -> str:
        """ print the node as a string """
        if self.linear_a == 1:
            if self.linear_b > 0:
                return "add({}, {:.3f})".format(self.tree.to_string(), self.linear_b)
            elif self.linear_b < 0:
                return "sub({}, {:.3f})".format(self.tree.to_string(), -1*self.linear_b)
            else:
                return "{}".format(self.tree.to_string())
        else:
            if self.linear_b > 0:
                return "add(mul({:.3f}, {}), {:.3f})".format(self.linear_a, self.tree.to_string(), self.linear_b)
            elif self.linear_b < 0:
                return "sub(mul({:.3f}, {}), {:.3f})".format(self.linear_a, self.tree.to_string(), -1*self.linear_b)
            else:
                return "mul({:.3f}, {})".format(self.linear_a, self.tree.to_string())

    def size(self) -> int:
        """ calc the size of the equation """
        return self.tree.size()

    def populate(self,
                 not_leaf_values: list,
                 leaf_values: list) -> list:
        """ provide a list with all possible combinations """
        # set index to all and get which one is leaf and not leaf
        leaf_dict = {}
        self.tree.set_index(leaf_dict)
        possible_allocations_list = [leaf_values if is_leaf else not_leaf_values
                                     for index, is_leaf in leaf_dict.items()]
        possible_allocations_index_list = [len(val) for val in possible_allocations_list]
        combinations_count = np.prod(possible_allocations_index_list)
        # run on all possible permutations
        answer = []
        for index in range(combinations_count):
            allocation_option = [0 for _ in range(len(possible_allocations_index_list))]
            current_index = index
            set_index = 0
            while current_index != 0:
                this_val = current_index % possible_allocations_index_list[set_index]
                allocation_option[set_index] = this_val
                current_index = current_index // possible_allocations_index_list[set_index]
                set_index += 1
            answer.append(Eq(tree=self.tree._copy_and_put_values(allocation={index: possible_allocations_list[index][val] for index, val in enumerate(allocation_option)})))
        return answer

    @staticmethod
    def all_possible_fbt(n: int) -> list:
        """ Return all full binary trees of inputted size 'n' """
        return [Eq(tree=eq) for eq in EqNode.all_possible_fbt(n=n)]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Eq: {:.2f}*{}+{:.2f}>".format(self.linear_a,
                                               self.tree.__str__(),
                                               self.linear_b)
