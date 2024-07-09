import pickle
from functools import wraps

import numpy as np
import torch.distributed as dist
from pathlib import Path
import random


def save_arguments(fn):
    """
    Save the parameters, use them as cache
    """

    @wraps(fn)
    def wrapped(*args, **kwargs):
        pickle.dump((args, kwargs), (Path("debug_cache") / fn.__name__).open("wb"))
        return fn(*args, **kwargs)

    return wrapped


def load_last_arguments(fn):
    @wraps(fn)
    def wrapped():
        args, kwargs = pickle.load((Path("debug_cache") / fn.__name__).open("rb"))
        return fn(*args, **kwargs)

    return wrapped


# this function will run if no cache is found, and load from disk if cache is found
def cached_return_value_for_debug(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        path = cache_dir / "debug_cache" / fn.__name__
        if path.exists():
            res = pickle.load(path.open("rb"))
            return res
        else:
            res = fn(*args, **kwargs)
            path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(res, path.open("wb"))
            return res

    return wrapped


def plot_image_tensor(tensor):
    array = tensor.detach().cpu().numpy()
    array = np.transpose(array, (1, 2, 0))
    plt.imshow(array, cmap='gray')
    plt.show()
    # plt.close()


def discretize(predicates):
    p = predicates.clone().detach()
    p[p > 0.5] = 1
    p[p <= 0.5] = 0
    return p


def binary_accuracy(logit_output, target, avg=True):
    assert logit_output.shape == target.shape
    logit_output = (logit_output > 0).long()
    ret = logit_output * target + (1 - logit_output) * (1 - target)
    if avg:
        ret = ret.mean()
        return ret.item()
    else:
        return ret


def accuracy(output, target, raw=False):
    if len(output.shape) == 1:
        pred = output > 0.5  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred))
        if raw:
            return correct
        else:
            correct = correct.float().mean().item()
            return correct
    else:
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred))
        if raw:
            return correct
        else:
            correct = correct.float().mean().item()
            return correct


def is_first_rank():
    try:
        world_size = dist.get_world_size()
    except RuntimeError:
        return True
    return dist.get_rank() == 0


def make_args_string_from_dict(args):
    ret = ""
    for k, v in args.items():
        ret += f"--{k} {v} "
    return ret


def set_choose_k_exclude(a_set, exclude_set, k):
    if len(a_set) < len(exclude_set) + k:
        raise RuntimeError("The list to be sampled from is too short")
    difference = a_set.difference(exclude_set)
    good_items = random.sample(list(difference), k)
    return good_items
