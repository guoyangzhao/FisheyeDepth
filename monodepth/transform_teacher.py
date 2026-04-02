import fire
import torch
from collections import OrderedDict


def transform_teacher_model(src_model_path:str,
                            tar_model_path:str):
    state_dict = torch.load(src_model_path, map_location='cpu')
    new_state_dict = OrderedDict()

    model_state_dict = state_dict['model_state_dict']
    for key in model_state_dict:
        if key.startswith('depth_backbone'):
            new_state_dict[key] = model_state_dict[key]
            continue
        if key.startswith('head.pose'):
            continue
        if key.startswith('head.depth_decoder'):
            new_key = key[5:]
            new_state_dict[new_key] = model_state_dict[key]
            continue

    torch.save(new_state_dict, tar_model_path)


if __name__ == '__main__':
    fire.Fire(transform_teacher_model)
