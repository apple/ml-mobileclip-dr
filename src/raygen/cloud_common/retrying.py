#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""Simple module to help with retrying and repeated execution."""

import functools
from typing import Any, Callable, Optional, Tuple, Type, TypeVar, cast

from .backoff_strategies import (
    BackOffFactoryType,
    BackoffStrategy,
    FixedTimeBackoffStrategy,
)

# Type aliases for readability
ExcHandlerFuncType = Callable[[Exception], None]
TryCallbackType = Callable[[], None]
ExceptionListType = Tuple[Type[Exception], ...]

CallableType = TypeVar("CallableType", bound=Callable[..., Any])


def retry(
    on_exceptions: ExceptionListType,
    backoff_factory: BackOffFactoryType = FixedTimeBackoffStrategy,
    exc_handler: Optional[ExcHandlerFuncType] = None,
    try_callback: Optional[TryCallbackType] = None,
) -> Callable[[CallableType], CallableType]:
    """A decorator that lets you wrap any function and retry its execution.

    A backoff strategy can be specified via the backoff_factory argument. It will
    overwrite tthe simpler fixed time backoff strategy that is the default.

    Args:
        on_exceptions: Specifies which Exception type(s) trigger a
                                                    retry
        backoff_factory: A factory method for custom back-off
                strategies. Should return an instance of type BackoffStrategy
        exc_handler: Optional exception handler, which
            implements additional behavior to decide if a retry should be done
        try_callback: A function that is called each time a retry-wrapped function is called

    Returns:
        A parametrized decorator that takes a function and returns a wrapped version with retry

    """

    def decorator(func: CallableType) -> CallableType:
        @functools.wraps(func)
        def func_wrapper(*args: Any, **kwargs: Any) -> Any:
            backoff = backoff_factory()
            while True:
                try:
                    if try_callback:
                        try_callback()
                    return func(*args, **kwargs)
                except on_exceptions as exc:
                    _handle_run_exception(exc, backoff, exc_handler)

        return cast(CallableType, func_wrapper)

    return decorator


def _handle_run_exception(
    exc: Exception,
    backoff: BackoffStrategy,
    exc_handler: Optional[Callable] = None,
) -> None:
    """Handle the raised exception with the given backoff and exception handler."""
    if exc_handler:
        try:
            exc_handler(exc)
        except Exception:
            backoff.logger.warning(
                "Exception of type: %s and content: '%s' "
                "is not covered by the custom exception handler.",
                type(exc),
                str(exc),
            )
            raise exc

    backoff.logger.warning(
        "Retrying on exception of type: %s with content '%s'", type(exc), str(exc)
    )
    success = backoff()
    if not success:
        raise exc
