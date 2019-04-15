import string
import random


def generate_random_string(length=10):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def generate_random_digits(length=10):
    return "".join(random.choice(string.digits) for _ in range(length))
