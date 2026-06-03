class BasePrecomputeHook(object):
    """
        Precomputing functions do not have input/output arguments.
        But they can have initialization parameters.
    """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass
