# library import
from sympy import *
import re as regular_exp


class SymbolicRegressionToLatexText:
    """
    This class is responsible to convert the standard symbolic regression's result style to latex style
    """

    # CONSTS #
    SR_FUNCS_NAMES = ["add", "sub", "mul", "div"]

    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def run(eq: str):
        """
        Single entry point - run the convertor from EQ of the symbolic regression class to LATEX style
        :param eq: the EQ to convert
        :return: the same EQ in LATEX format
        """
        # replace the text to use the static methods of this class
        for func_name in SymbolicRegressionToLatexText.SR_FUNCS_NAMES:
            eq = eq.replace(func_name,
                            "SymbolicRegressionToLatexText._{}".format(func_name))
        eq = eq.replace("^", "power")
        # collect possible var names
        eq_vars = regular_exp.findall(r'(\w*),',
                                      eq)
        eq_vars.extend(regular_exp.findall(r', (\w*)\)',
                                           eq))
        # filter just vars names
        eq_vars = list(set([eq_var.strip() for eq_var in eq_vars if len(eq_var) > 0 and not eq_var.isnumeric() and eq_var not in SymbolicRegressionToLatexText.SR_FUNCS_NAMES]))
        # name them as strings
        eq_vars = sorted(eq_vars,
                         key=lambda x: len(x),
                         reverse=True)
        for eq_var in eq_vars:
            eq = eq.replace("{},".format(eq_var), '"{}",'.format(eq_var))
            eq = eq.replace(", {}".format(eq_var), ', "{}"'.format(eq_var))
        # run the code
        ex_locals = {}
        exec("answer = {}".format(eq), None, ex_locals)
        answer_eq = ex_locals["answer"]
        # small fixes to style
        try:
            answer_eq = SymbolicRegressionToLatexText._post_fixes(answer_eq=answer_eq)
        except:
            pass
        return answer_eq

    @staticmethod
    def _add(x: str,
             y: str):
        x = str(x)
        y = str(y)
        x_number = x.isnumeric() or (x[1:].isnumeric() and x[0] == "-")
        y_number = y.isnumeric() or (y[1:].isnumeric() and y[0] == "-")
        if x_number and y_number:
            return "{}".format(float(x) + float(y))
        return "({} + {})".format(x, y)

    @staticmethod
    def _sub(x: str,
             y: str):
        x = str(x)
        y = str(y)
        x_number = x.isnumeric() or (x[1:].isnumeric() and x[0] == "-")
        y_number = y.isnumeric() or (y[1:].isnumeric() and y[0] == "-")
        if x_number and y_number:
            return "{}".format(float(x) - float(y))
        return "({} - {})".format(x, y)

    @staticmethod
    def _mul(x: str,
             y: str):
        x = str(x)
        y = str(y)
        x_number = x.isnumeric() or (x[1:].isnumeric() and x[0] == "-")
        y_number = y.isnumeric() or (y[1:].isnumeric() and y[0] == "-")
        if x_number and y_number:
            return "{}".format(float(x) * float(y))
        elif x_number and not y_number:
            return "{}{}".format(x, y)
        elif not x_number and y_number:
            return "{}{}".format(y, x)
        else:
            return "{} \\cdot {}".format(x, y)

    @staticmethod
    def _div(x: str,
             y: str):
        x = str(x)
        y = str(y)
        x_number = x.isnumeric() or (x[1:].isnumeric() and x[0] == "-")
        y_number = y.isnumeric() or (y[1:].isnumeric() and y[0] == "-")
        if x_number and y_number:
            return "{}".format(float(x) / float(y))
        else:
            return "\\frac{" + str(x) + "}{" + str(y) + "}"

    @staticmethod
    def _post_fixes(answer_eq: str):
        change_symbol = True
        while change_symbol:
            answer_eq_before = answer_eq
            answer_eq = answer_eq.replace("--", "+")
            answer_eq = answer_eq.replace(" - -", " + ")
            answer_eq = answer_eq.replace("-+", "-")
            answer_eq = answer_eq.replace("+-", "-")
            answer_eq = answer_eq.replace("++", "+")
            answer_eq = answer_eq.replace(" + +", "+")
            answer_eq = answer_eq.replace(" - +", " - ")
            answer_eq = answer_eq.replace(" + -", " - ")
            answer_eq = answer_eq.replace("power", "^")
            change_symbol = answer_eq_before != answer_eq
        # try to simplify results
        try:
            answer_eq = simplify(answer_eq)
        except Exception as error:
            pass
        # return answer
        return answer_eq
