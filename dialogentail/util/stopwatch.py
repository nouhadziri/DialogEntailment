from time import time


class Stopwatch:
    def __init__(self):
        self.start()

    def start(self):
        self.__start = time()

    def elapsed(self):
        return round(time() - self.__start, 3)


def profile(func):
    def wrapper(*args, **kwargs):
        sw = Stopwatch()
        func(*args, **kwargs)
        print(f"{func.__name__} done in {sw.elapsed():.1f}s")

    return wrapper
