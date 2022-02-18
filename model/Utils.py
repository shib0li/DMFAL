import torch

def Kronecker(A, B):
    AB = torch.einsum("ab,cd->acbd", A, B)
    AB = AB.view(A.size(0)*B.size(0), A.size(1)*B.size(1))
    return AB