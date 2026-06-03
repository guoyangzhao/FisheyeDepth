import torchvision

def densenet(depth, **kwargs):
    if 'pretrained' in kwargs:
        pretrained = kwargs['pretrained']
    else:
        pretrained = True
    if depth == 121:
        model = torchvision.models.densenet121(pretrained=pretrained, **kwargs).features
    elif depth == 161:
        model = torchvision.models.densenet161(pretrained=pretrained, **kwargs).features
    elif depth == 169:
        model = torchvision.models.densenet169(pretrained=pretrained, **kwargs).features
    elif depth == 201:
        model = torchvision.models.densenet201(pretrained=pretrained, **kwargs).features
    else:
        raise ValueError(
            'Unsupported model depth, must be one of 121, 161, 169, 201')
    del model.transition3.pool
    return model
