import torch.nn as nn

class BaseMetaArch(nn.Module):
    def forward_train(self, data, meta):
        raise NotImplementedError
        return_dict = dict(
            loss=0,
            loss_dict=dict(),
        )
        return return_dict

    def forward_test(self, data, meta):
        raise NotImplementedError
        return dict()
    
    def dummy_forward(self, data):
        return dict()

    def forward(self, data, meta):
        if meta['is_training']:
            return self.forward_train(data, meta)
        else:
            return self.forward_test(data, meta)
