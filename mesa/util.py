"""Utilities used across mesa."""

import warnings
from functools import wraps


def deprecate_kwarg(name: str):
    """Deprecation warning wrapper for specified kwarg."""

    def inner_wrapper(method):
        """Deprecation warning wrapper for seed kwarg."""

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            """Inner function."""
            if kwargs.get(name) is not None:
                warnings.warn(
                    f"The use of {name} is deprecated, please use rng instead",
                    FutureWarning,
                    stacklevel=2,
                )

                if kwargs.get("rng") is not None:
                    raise ValueError(f"you have to pass either rng or {name}, not both")

            return method(self, *args, **kwargs)

        return wrapper

    return inner_wrapper
