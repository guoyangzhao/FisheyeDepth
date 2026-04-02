import fire
import numpy as np
import time
from synology_api import filestation
import getpass
import warnings
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from _path_init import *
from vision_base.utils.utils import cfg_from_file

def find_latest_modified_path(list_of_dir):
    list_time = [os.stat(folder).st_mtime for folder in list_of_dir]
    sorted_index = np.argsort(list_time)
    return list_of_dir[sorted_index[-1]]

def datestring_from_wall_time(walltime):
    iso = time.strftime('%Y-%m-%dT%H_%M_%SZ', time.localtime(walltime))
    return iso
        

def try_login_once(username, password, IP="nas.ram-lab.com", port="5000"):
    try:
        fl = filestation.FileStation(IP, port, username, password,
                                    secure=False, cert_verify=False, dsm_version=7,
                                    debug=True, otp_code=None)
        return fl
    except KeyError:
        warnings.warn("Log in fail")
        return None

def login():
    username = input("NAS username: ")
    password = getpass.getpass("NAS password: ")
    result = try_login_once(username, password)

    if result is None:
        print("Please re-enter user name and password")
        username = input("NAS username: ")
        password = getpass.getpass("NAS password: ")
        result = try_login_once(username, password)
        if result is None:
            print("Login failed twice, exit program")
            exit()
    return_dict = dict(
        fl=result,
        username = username,
    )
    return return_dict

def fix_string_from_TensorEvent(tensor_event):
    return tensor_event.tensor_proto.string_val[0].decode("utf-8").replace("&nbsp;", " ")

def main(config_path, experiment_name='default'):
    cfg = cfg_from_file(config_path)
    log_path = cfg.path.log_path
    tb_path = os.path.join(log_path, f"{experiment_name}config={os.path.basename(config_path)}")
    if os.path.isdir(tb_path):
        ea = EventAccumulator(tb_path)
    else:
        print(f"{tb_path} is not found")
        list_of_experiments = os.listdir(log_path)
        tb_path = find_latest_modified_path(
            [os.path.join(log_path, experiment) for experiment in list_of_experiments]
        )
        print(f"Selected {tb_path} from {list_of_experiments}")
        ea = EventAccumulator(tb_path)
    ea.Reload()

    text_result_tags = ea.Tags()['tensors']
    evaluation_tags = [tag for tag in text_result_tags
                if 'evaluation' in tag.lower() or 'validation' in tag.lower() or 'test' in tag.lower()]
    if len(evaluation_tags) == 0:
        print("Not evaluated yet")
        exit()
    
    assert "config.py/text_summary" in text_result_tags
    assert "git/git_show/text_summary" in text_result_tags
    assert "model structure/text_summary" in text_result_tags

    config_file_name = "expanded_config.py"
    git_file_name = "git_diff.md"
    model_file_name = "model.txt"

    config_time = ea.Tensors("config.py/text_summary")[0].wall_time
    iso_time_string = datestring_from_wall_time(config_time)
    config_string = fix_string_from_TensorEvent(ea.Tensors("config.py/text_summary")[0])
    
    
    git_diff_str = fix_string_from_TensorEvent(ea.Tensors("git/git_show/text_summary")[0])
    

    model_str = fix_string_from_TensorEvent(ea.Tensors("model structure/text_summary")[0])
    

    step = 0
    result_string = ""
    for tag in evaluation_tags:
        event = ea.Tensors(tag)[-1]
        string = fix_string_from_TensorEvent(event)
        result_string += f"{tag}\n" + string + "\n"
        step = max(event.step, step)

    checkpoint_dir = cfg.path.checkpoint_path
    model_name = cfg.meta_arch.name
    checkpoint_path = os.path.join(
        checkpoint_dir, f"{model_name}_{step}.pth"
    )

    
    result = login()
    fl = result['fl']
    username = result['username']
    base_dir = f"/data/{username}/monodepth"
    filename = config_path.split("/")[-1]
    this_folder = f"{filename}_{experiment_name}_{iso_time_string}"
    test_dir_path = os.path.join(base_dir, this_folder)
    print(base_dir, this_folder)
    fl.create_folder(base_dir, this_folder)

    ## Config
    with open(config_file_name, 'w') as f:
        f.write("from numpy import array\nfrom easydict import EasyDict as edict\ncfg=edict(")
        f.write(config_string + ")")
    fl.upload_file(test_dir_path, config_file_name)
    os.remove(config_file_name)
    print("Uploaded Config file")

    ## Git
    with open(git_file_name, 'w') as f:
        f.write(git_diff_str)
    fl.upload_file(test_dir_path, git_file_name)
    os.remove(git_file_name)
    print("Uploaded Git file")

    ## Model
    with open(model_file_name, 'w') as f:
        f.write(model_str)
    fl.upload_file(test_dir_path, model_file_name)
    os.remove(model_file_name)
    print("Uploaded Model file")

    ## Checkpoint
    fl.upload_file(test_dir_path, checkpoint_path)
    print("Uploaded Checkpoint file")

    ## tensorboard
    tb_name = os.listdir(tb_path)[0]
    fl.upload_file(test_dir_path, os.path.join(tb_path, tb_name))
    print("Uploaded Tensorboard file")


fire.Fire(main)
