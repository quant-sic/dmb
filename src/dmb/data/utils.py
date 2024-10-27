"""Utility functions for data processing."""

from functools import reduce
from typing import Callable, Iterable

from dmb.logging import create_logger

log = create_logger(__name__)


def chain_fns(functions: Iterable[Callable]) -> Callable:
    """Chain multiple functions together.

    Args:
        fns: Iterable of functions to chain together.
    
    Returns:
        Callable: A function that chains the input functions together.
    """
    return reduce(lambda f, g: lambda x: g(f(x)), functions)
