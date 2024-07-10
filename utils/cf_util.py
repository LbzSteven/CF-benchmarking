import torch.multiprocessing as mp
import time
import signal
import functools

class TimeoutException(Exception):
    pass
def timeout_handler(signum, frame):
    raise TimeoutException

def timeout(seconds=3600, default_value=None):
    if default_value is None:
        default_value = ['To', None]

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            except TimeoutException:
                print("Function timed out")
                return default_value
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator