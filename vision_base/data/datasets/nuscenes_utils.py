GLOBAL_DICT = {}
def NuScenes(dataroot, version, *args,  **kwargs):
    if (dataroot, version) not in GLOBAL_DICT:
        from nuscenes.nuscenes import NuScenes as NuSceneObj
        GLOBAL_DICT[(dataroot, version)] = NuSceneObj(version=version, dataroot=dataroot, *args, **kwargs)
    return GLOBAL_DICT[(dataroot, version)]
