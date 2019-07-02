import torch
import numpy as np
from object_pose_utils.utils import to_np
from pybingham import bingham_F, bingham_dF

class BinghamConst(torch.autograd.Function):
    """
    Pytorch Bingham normalization constant function.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        #F = bingham_F(input.detach().numpy().astype(np.double))
        F = bingham_F(to_np(input).astype(np.double))
        
        return torch.as_tensor(F, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        #Z = input.detatch().numpy()
        Z = to_np(input)
        dF = bingham_dF(Z)
        # Not sure if this always prevents the NANs? Need to check
        #z_i = np.argsort(Z)
        #dF = np.array(bingham_dF(Z[z_i]))
        #dF[z_i] = dF
        if(np.any(np.isnan(dF))):
            print('BinghamConst: Gradient NaN')
            dF = np.zeros_like(dF)
        grad_input = grad_output.clone() * torch.as_tensor(dF, dtype=grad_output.dtype) 
        if(torch.cuda.is_available()):
            grad_input = grad_input.cuda()
        #grad_input *= torch.as_tensor(dF, dtype=grad_output.dtype)
        return grad_input 

def bingham_const(input): 
    return BinghamConst().apply(input)

