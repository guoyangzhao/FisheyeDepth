import numpy as np
import torch
import sys
import os
import tempfile
import shutil
import importlib
import random
from easydict import EasyDict
try:
    from torch import _assert as torch_assert
except ImportError:
    def torch_assert(condition: bool, message: str):
        assert condition, message


def get_num_parameters(model):
    """Count number of trained parameters of the model"""
    if hasattr(model, 'module'):
        num_parameters = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    else:
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return num_parameters

def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def cfg_from_file(cfg_filename:str)->EasyDict:
    assert cfg_filename.endswith('.py')

    with tempfile.TemporaryDirectory() as temp_config_dir:
        temp_config_file = tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix='.py')
        temp_config_name = os.path.basename(temp_config_file.name)
        shutil.copyfile(cfg_filename, os.path.join(temp_config_dir, temp_config_name))
        temp_module_name = os.path.splitext(temp_config_name)[0]
        sys.path.insert(0, temp_config_dir)
        cfg = getattr(importlib.import_module(temp_module_name), 'cfg')
        assert isinstance(cfg, EasyDict)
        sys.path.pop(0)
        del sys.modules[temp_module_name]
        temp_config_file.close()

    return cfg


def update_dict(obj:dict, key:str, rest_items:list, value):
    """Update the configuration dictionary iteratively.
        1. if the key is the last key name, directly assign the new value and return.
        2.0. if the key not exist in obj or not a dict, we make it a dictionary. and move on to three.
        2.1. we update the corresponding key with the value and the other keys.

        Example:
            obj = {'a':1,
                   'b':{'c':0},
                  }
            obj = update_dict(obj, 'a', [], value)
            print(obj) # obj['a'] = value
            obj = update_dict(obj, 'b', ['c'], value)
            print(obj) # obj['b']['c'] = value
            obj = update_dict(obj, 'd', ['e', 'f'], value)
            print(obj) # obj['d']['e']['f'] = value;
    """
    if len(rest_items) == 0:
        obj[key] = value
        return obj

    if not (key in obj and isinstance(obj[key], dict)):
        obj[key] = EasyDict()
    obj[key] = update_dict(obj[key], rest_items[0], rest_items[1:], value)
    return obj

def update_cfg(cfg:EasyDict, **kwargs:dict):
    """Update configuration with commandline key inputs

    Example:
        cfg = {'a':1,
               'b':{'c':0, 'f':2},
               'c':3
              }
        kwargs={
            'a' : 2,
            'b.c' : 3,
            'd.e.f' : 4,
            'c.g' : 1
        }
        cfg = update_cfg(cfg, **kwargs)
        # original
        assert(cfg['b']['f'] == 2)
        # direct updates
        assert(cfg['a'] == 2)
        assert(cfg['b']['c'] == 3)
        # Create new sub dictionary
        assert(isinstance(cfg['d']['e'], dict))
        assert(cfg['d']['e']['f'] == 4)
        # Overwrite datatype of existing value
        assert(isinstance(cfg['c'], dict))
        assert(cfg['c']['g'] == 1)
    """
    for key in kwargs:
        value = kwargs[key]
        key_items = key.split('.')
        cfg=update_dict(cfg, key_items[0], key_items[1:], value)
    return cfg

def merge_name(list_of_name):
    """
        Merge ['A', 'B', 'C'] to 'A.B.C' with '.' seperation.
    """
    final_name = ""
    for name in list_of_name:
        final_name += name + "."

    final_name = final_name.strip('.')
    return final_name


def find_object(object_string:str):
    """Return the object(module, class, function) searching with string.
    Args:
        object_string (str)
    Return:
        module/class/function, None with not found
    Example:
    1.
        import torch
        torch_module = find_object('torch')
        torch_module.sigmoid == torch.sigmoid
    2.
        exp = find_object('numpy.exp')
        e = exp(1.0)
    """

    function_name = object_string
    splitted_names = function_name.split('.')

    is_found = False
    error_traces = []
    for i in range(len(splitted_names), 0, -1):
        try:
            merged_name = merge_name(splitted_names[0:i])
            module = importlib.import_module(merged_name)
            base_obj = module
            for name in splitted_names[i:]:
                base_obj = getattr(base_obj, name)
        except Exception as e:
            error_traces.append((merged_name, e))
            continue
            
        is_found = True
        break
    
    if is_found:
        return base_obj
    
    else:
        error_log = ""
        for name, error in error_traces:
            error_log = error_log + f"{name} : {error} \n"
        raise ModuleNotFoundError(f"{object_string} not imported, error traces: \n{error_log}")
