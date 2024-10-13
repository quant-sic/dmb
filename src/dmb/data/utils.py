"""Utility functions for data processing."""

from functools import reduce
from typing import Iterable

from dmb.logging import create_logger

log = create_logger(__name__)


def chain_fns(functions: Iterable[callable]) -> callable:
    """Chain multiple functions together.

    Args:
        fns: Iterable of functions to chain together.
    
    Returns:
        callable: A function that chains the input functions together.
    """
    return reduce(lambda f, g: lambda x: g(f(x)), functions)
