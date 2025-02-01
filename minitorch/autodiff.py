from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol
from collections import defaultdict

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # Convert the tuple to a list to modify it
    vals_list_plus = list(vals)
    vals_list_minus = list(vals)

    # Modify the `arg`-th element by adding and subtracting epsilon
    vals_list_plus[arg] += epsilon
    vals_list_minus[arg] -= epsilon

    # Convert the lists back to tuples
    vals_plus_eps = tuple(vals_list_plus)
    vals_minus_eps = tuple(vals_list_minus)

    # Apply the central difference formula
    forward_diff = (f(*vals_plus_eps) - f(*vals_minus_eps)) / (2 * epsilon)

    return forward_diff


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    order = []
    seen = set()

    def dfs(var):
        if var.unique_id in seen:
            return
        if not var.is_leaf():
            for inp in var.history.inputs:
                dfs(inp)
        
        seen.add(var.unique_id)
        order.insert(0, var)

    dfs(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    sorted_variables = topological_sort(variable=variable)
    deriv_dict = defaultdict(float)
    deriv_dict[variable.unique_id] = deriv
    for scalar in sorted_variables:
        d = deriv_dict[scalar.unique_id]
        # if scalar is not a leaf we pass along the derivative to the parents according to chain rule
        if not scalar.is_leaf():
            for parent, grad in scalar.chain_rule(d):
                deriv_dict[parent.unique_id] += grad
        else:
            scalar.accumulate_derivative(d)



@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
