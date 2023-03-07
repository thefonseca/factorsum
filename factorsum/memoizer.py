from dataclasses import dataclass
from functools import wraps
import inspect
import logging
import os
from typing import Optional, List

import numpy as np
from diskcache import Cache

from datasets.fingerprint import update_fingerprint
import fire


@dataclass(frozen=True)
class Constants:
    CACHE_MISS = 0


constants = Constants()
logger = logging.getLogger(__name__)
default_cache = Cache(
    os.path.join(os.getenv("HOME", "."), ".memoize"), size_limit=int(40e9)
)


def _call_func(func, self, *args, **kwargs):
    if self:
        out = func(self, *args, **kwargs)
    else:
        out = func(*args, **kwargs)
    return out


def _get_item(cache_provider, key, func, self, expire, ignore_cache, *args, **kwargs):
    with Cache(cache_provider.directory) as cache:
        result = cache.get(key, default=constants.CACHE_MISS, retry=True)
        cache_hit = result != constants.CACHE_MISS

        if ignore_cache or not cache_hit:
            result = _call_func(func, self, *args, **kwargs)
            cache.set(key, result, expire=expire, retry=True)
        else:
            logger.debug(f"Item from cache: {key}")
    return result, cache_hit


def _remove_default_kwargs(func, kwargs):
    # kwargs = kwargs.copy()
    default_values = {
        p.name: p.default
        for p in inspect.signature(func).parameters.values()
        if p.default != inspect._empty
    }

    for default_varname, default_value in default_values.items():
        if default_varname in kwargs and kwargs[default_varname] == default_value:
            kwargs.pop(default_varname)


def _add_seed(kwargs):
    # kwargs = kwargs.copy()
    if kwargs.get("seed") is None and kwargs.get("generator") is None:
        _, seed, pos, *_ = np.random.get_state()
        seed = seed[pos] if pos < 624 else seed[0]
        kwargs["generator"] = np.random.default_rng(seed)


def _filter_kwargs(kwargs, use_kwargs, ignore_kwargs):
    if use_kwargs:
        kwargs = {k: v for k, v in kwargs.items() if k in use_kwargs}
    if ignore_kwargs:
        kwargs = {k: v for k, v in kwargs.items() if k not in ignore_kwargs}
    return kwargs


def memoize(
    cache=None,
    ignore_cache=False,
    use_kwargs: Optional[List[str]] = None,
    ignore_kwargs: Optional[List[str]] = None,
    randomized_function: bool = False,
    version: Optional[str] = None,
    expire: Optional[float] = None,
    log_level: Optional[int] = logging.DEBUG,
):
    if cache is None:
        cache = default_cache

    if use_kwargs is not None and not isinstance(use_kwargs, list):
        raise ValueError(f"use_kwargs is supposed to be a list, not {type(use_kwargs)}")

    if ignore_kwargs is not None and not isinstance(ignore_kwargs, list):
        raise ValueError(
            f"ignore_kwargs is supposed to be a list, not {type(use_kwargs)}"
        )

    def _memoize(func):

        if (
            randomized_function
        ):  # randomized function have seed and generator parameters
            if "seed" not in func.__code__.co_varnames:
                raise ValueError(f"'seed' must be in {func}'s signature")
            if "generator" not in func.__code__.co_varnames:
                raise ValueError(f"'generator' must be in {func}'s signature")

        # this has to be outside the wrapper or since __qualname__ changes in multiprocessing
        transform = f"{func.__module__}.{func.__qualname__}"
        if version is not None:
            transform += f"@{version}"

        memoize_info = dict(ignore_cache=ignore_cache)

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal memoize_info
            kwargs_for_fingerprint = kwargs.copy()

            if args:
                params = [
                    p.name
                    for p in inspect.signature(func).parameters.values()
                    if p != p.VAR_KEYWORD
                ]
                kwargs_for_fingerprint.update(zip(params, args))

            self = None
            if (
                "self" in kwargs_for_fingerprint
                and kwargs_for_fingerprint["self"] == args[0]
            ):
                self = kwargs_for_fingerprint.pop("self")
                args = args[1:]
                params = params[1:]

            kwargs_for_fingerprint = _filter_kwargs(
                kwargs_for_fingerprint, use_kwargs, ignore_kwargs
            )

            # If ignore_cache kwarg is provided, override decorator value
            ignore_cache_kwarg = kwargs_for_fingerprint.pop(
                "memoizer_ignore_cache", None
            )
            if ignore_cache_kwarg is not None:
                memoize_info["ignore_cache"] = ignore_cache_kwarg

            if randomized_function:
                # randomized functions have `seed` and `generator` parameters
                _add_seed(kwargs_for_fingerprint)

            # remove kwargs that are the default values
            _remove_default_kwargs(func, kwargs_for_fingerprint)

            fingerprint = update_fingerprint(None, transform, kwargs_for_fingerprint)
            out = None
            if cache is not None:
                out, cache_hit = _get_item(
                    cache,
                    fingerprint,
                    func,
                    self,
                    expire,
                    memoize_info["ignore_cache"],
                    *args,
                    **kwargs,
                )
            else:
                out = _call_func(func, self, *args, **kwargs)

            memoize_info["fingerprint"] = fingerprint
            memoize_info["cache_hit"] = cache_hit

            if cache_hit:
                logger.log(
                    log_level,
                    f"Loaded {transform} result from cache (fingerprint: {memoize_info['fingerprint']})",
                )
            return out

        wrapper._decorator_name_ = "memoize"
        wrapper.memoize_info = memoize_info
        return wrapper

    return _memoize


@memoize()
def test(a, b):
    return a + b


def main(a=1, b=2):
    result = test(a, b)
    print("Memoization info:", test.memoize_info)
    print(result)


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    fire.Fire(main)
