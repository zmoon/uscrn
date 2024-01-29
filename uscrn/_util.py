import logging

logger = logging.getLogger("uscrn")


def retry(func):
    """Decorator to retry a function on web connection error.
    Up to 60 s, with Fibonacci backoff (1, 1, 2, 3, ...).
    """
    import urllib
    from functools import wraps

    import requests

    max_time = 60  # seconds

    @wraps(func)
    def wrapper(*args, **kwargs):
        from time import perf_counter_ns, sleep

        t0 = perf_counter_ns()
        a, b = 1, 1
        while True:
            try:
                return func(*args, **kwargs)
            except (
                urllib.error.URLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
            ):
                if perf_counter_ns() - t0 > max_time * 1_000_000_000:
                    raise
                logger.info(
                    f"Retrying {func.__name__} in {a} s after connection error",
                    stacklevel=2,
                )
                sleep(a)
                a, b = b, a + b  # Fibonacci backoff

    return wrapper
