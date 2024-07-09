from pathlib import Path
import pickle


def project_root():
    return Path(__file__).absolute().parent.parent


def debug_dump(obj):
    with open(Path.home() / 'nsexec_debug.pkl', 'wb') as f:
        pickle.dump(obj, f)


def debug_load():
    with open(Path.home() / 'nsexec_debug.pkl', 'rb') as f:
        obj = pickle.load(f)
        return obj


class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value



from pathlib import Path
import pickle


class PickleCacheTool:
    def __init__(self):
        self.cache_dir = project_root() / "cache"

    def save(self, name, payload):
        with open(self.cache_dir / name, 'wb') as f:
            pickle.dump(payload, f)

    def load(self, name):
        with open(self.cache_dir / name, 'rb') as f:
            return pickle.load(f)

    def cache_exists(self, name):
        return (self.cache_dir / name).exists()
