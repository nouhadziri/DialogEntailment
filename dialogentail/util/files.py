import os
import pickle
from smart_open import smart_open


def _split3(path):
    dir, f = os.path.split(path)
    fname, ext = os.path.splitext(f)

    return dir, fname, ext


def get_containing_dir(path):
    d, _, _ = _split3(path)
    return d


def get_parent_dir(path):
    if os.path.isfile(path):
        path = get_containing_dir(path)
    return os.path.abspath(os.path.join(path, os.pardir))


def get_file_name(path):
    _, fname, _ = _split3(path)
    return fname


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with smart_open(name, 'rb') as f:
        return pickle.load(f)
