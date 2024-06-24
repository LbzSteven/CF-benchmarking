import multiprocessing
import time

def timeout(seconds=3600, default_value=None):
    if default_value is None:
        default_value = ['To', None]
    def decorator(func):
        def wrapper(*args, **kwargs):
            def target(queue, *args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    queue.put(result)
                except Exception as e:
                    queue.put(e)

            queue = multiprocessing.Queue()
            process = multiprocessing.Process(target=target, args=(queue, *args), kwargs=kwargs)
            process.start()
            process.join(seconds)
            if process.is_alive():
                process.terminate()
                process.join()
                return default_value
            result = queue.get()
            if isinstance(result, Exception):
                return default_value
            return result

        return wrapper

    return decorator