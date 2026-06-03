from typing import List, Callable, Dict
import numpy as np
from .utils import find_object

def build(name, *args, **kwargs):
    obj = find_object(name)
    return obj(*args, **kwargs)

class Sequential(object):
    """
        A set of classes runs in sequential. All the input_output should be directly fed in sequential.
    """
    def __init__(self, cfg_list:List[Dict], **common_keywords):
        self.children:List[Callable] = []
        for item in cfg_list:
            tmp = common_keywords.copy()
            tmp.update(item)
            self.children.append(build(**tmp))

    def __call__(self, *args, **kwargs):
        for i, child in enumerate(self.children):
            if i == 0:
                result = child(*args, **kwargs)
            else:
                if isinstance(result, tuple):
                    result = child(*result)
                else:
                    result = child(result)
        return result

class Parallel(object):
    """
        A set of classes runs in parallel. All the inputs are the same, all outputs are contat into a list.
    """
    def __init__(self, cfg_list:List[Dict], **common_keywords):
        self.children:List[Callable] = []
        for item in cfg_list:
            tmp = common_keywords.copy()
            tmp.update(item)
            self.children.append(build(**tmp))

    def __call__(self, *args, **kwargs):
        results = []
        for i, child in enumerate(self.children):
            results.append(child(*args, **kwargs))
        return results

class Shuffle(object):
    """
        A set of classes runs in random sequence. All the input_output should be directly fed in sequential.
    """
    def __init__(self, cfg_list:List[Dict], **common_keywords):
        self.children:List[Callable] = []
        for item in cfg_list:
            tmp = common_keywords.copy()
            tmp.update(item)
            self.children.append(build(**tmp))

    def __call__(self, *args, **kwargs):
        # We aim to keep the original order of the initialized transforms in self.transforms, so we only randomize the indexes.
        shuffled_indexes = np.random.permutation(len(self.children))

        for i, index in enumerate(shuffled_indexes):
            if i == 0:
                result = self.children[index](*args, **kwargs)
            else:
                if isinstance(result, tuple):
                    result = self.children[index](*result)
                else:
                    result = self.children[index](result)
        
        return result
