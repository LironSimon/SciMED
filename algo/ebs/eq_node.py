# library imports
import pandas as pd

# project imports
from algo.ebs.eq_functions import *


class EqNode:
    """
    This class represents a single node in an equation tree
    """

    # CONSTS #
    NO_INDEX = -1
    # END - CONSTS #

    # CATCH FOR OPTIMIZATION #
    dp = {}
    # END - CATCH FOR OPTIMIZATION #

    def __init__(self,
                 value=None,
                 is_leaf: bool = True,
                 left_child=None,
                 right_child=None,
                 index: int = NO_INDEX):
        self.value = value
        self.is_leaf = is_leaf
        self.index = index
        if is_leaf:
            self.left_child = None
            self.right_child = None
        elif left_child is not None and right_child is not None:
            self.left_child = left_child
            self.right_child = right_child
        else:
            raise ValueError("If EqNode is not leaf node, it must have both left and right childs")

    def eval(self,
             x_values: pd.DataFrame) -> pd.Series:
        """ eval the node """
        if self.is_leaf:
            return x_values[self.value]
        else:
            return self.value(self.left_child.eval(x_values),
                              self.right_child.eval(x_values))

    def fix_node(self) -> None:
        """ fix nodes' is_leaf flag if has been corrupted by other process """
        if self.is_leaf and self.left_child is not None and self.right_child is not None:
            self.is_leaf = False
        elif not self.is_leaf and self.left_child is None and self.right_child is None:
            self.is_leaf = True
        elif self.is_leaf and (self.left_child is None or self.right_child is None):
            self.left_child = None
            self.right_child = None

        if not self.is_leaf:
            self.left_child.fix_node()
            self.right_child.fix_node()

    def to_string(self) -> str:
        """ print the node as a string """
        if self.is_leaf:
            return str(self.value)
        else:
            return "{}({}, {})".format(FUNCTION_MAPPER[self.value],
                                       self.left_child.to_string(),
                                       self.right_child.to_string())

    def to_id_str(self) -> str:
        """ print equation in a narrow way for hash mapping """
        if self.is_leaf:
            return "L".format()
        return "{}N{}".format(self.left_child.to_id_str(),
                              self.right_child.to_id_str())

    def size(self) -> int:
        """ calc the size of the equation """
        if self.is_leaf:
            return 1
        return 1 + self.right_child.size() + self.left_child.size()

    def set_index(self,
                  leaf_dict: dict,
                  index: int = 0) -> tuple:
        """ add an index to each node and tells if leaf or not """
        self.index = index
        leaf_dict[self.index] = self.is_leaf
        if not self.is_leaf:
            index = self.left_child.set_index(leaf_dict=leaf_dict, index=index + 1)
            index = self.right_child.set_index(leaf_dict=leaf_dict, index=index + 1)
        return index

    def _copy_and_put_values(self,
                             allocation: dict):
        """ copy the current topoloy and puts values by order according to their index """
        if self.is_leaf:
            return EqNode(value=allocation[self.index],
                          index=self.index,
                          left_child=None,  # self.left_child
                          right_child=None,  # self.right_child
                          is_leaf=self.is_leaf)
        return EqNode(value=allocation[self.index],
                      index=self.index,
                      left_child=self.left_child._copy_and_put_values(allocation=allocation),
                      right_child=self.right_child._copy_and_put_values(allocation=allocation),
                      is_leaf=self.is_leaf)

    @staticmethod
    def all_possible_fbt(n: int) -> list:
        """ Return all full binary trees of inputted size 'n' """
        if n == 0:
            return []
        if n == 1:
            return [EqNode(is_leaf=True)]
        if n in EqNode.dp:
            return EqNode.dp[n]

        result = []
        for l in range(n):
            r = n - 1 - l
            left_trees = EqNode.all_possible_fbt(n=l)
            right_trees = EqNode.all_possible_fbt(n=r)
            for t1 in left_trees:
                for t2 in right_trees:
                    result.append(EqNode(is_leaf=False,
                                         left_child=t1,
                                         right_child=t2))
        EqNode.dp[n] = result
        return result

    def __repr__(self):
        return "<EqNode: value={}, is_leaf={}, index={}>".format(self.value,
                                                                 self.is_leaf,
                                                                 self.index)

    def __str__(self):
        if self.is_leaf:
            return "([#{}]{})".format(self.index,
                                      self.value)
        return "([#{}]{} -> {} & {})".format(self.index,
                                             self.value,
                                             self.left_child.__str__(),
                                             self.right_child.__str__())
