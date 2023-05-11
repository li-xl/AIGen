
import torch.optim as optim

from gen.utils.registry import OPTIMS

@OPTIMS.register_module()
class SGD(optim.SGD):
    def __init__(self, params, lr=1e-1, momentum=0.9, dampening=0.1,
                 weight_decay=1e-4, nesterov=False):
        super().__init__(params,lr,momentum,dampening,weight_decay,nesterov)

@OPTIMS.register_module()
class Adam(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-4, amsgrad=False):
        super().__init__(params,lr,betas,eps,weight_decay,amsgrad)

OPTIMS.register_module("AdamW",optim.AdamW)