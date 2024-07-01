import asyncio
from typing import Any, Coroutine, TypeVar


def get_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        return asyncio.new_event_loop()


T = TypeVar("T")


def asyncio_run(coro: Coroutine[Any, Any, T]) -> T:
    loop = get_event_loop()
    return loop.run_until_complete(coro)
