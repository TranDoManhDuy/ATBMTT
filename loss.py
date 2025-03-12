import torch
import torch.nn as nn
import hashlib

class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.mse = nn.MSELoss(reduction="mean")
    
    def sha1_hash(self, tensor):
        tensor_bytes = tensor.detach().cpu().numpy().astype('uint8').tobytes()
        hash_hex = hashlib.sha1(tensor_bytes).hexdigest()

        hash_array = torch.tensor([int(hash_hex[i:i+2], 16) for i in range(0, 40, 2)], dtype=torch.float32)
        return hash_array.requires_grad_()

    def forward(self, predictions, targets):
        hash_pred = self.sha1_hash(predictions).clone().detach().requires_grad_(True)
        hash_targ = self.sha1_hash(targets).clone().detach().requires_grad_(True)

        loss_hash = self.mse(hash_pred, hash_targ)
        loss_mse = self.mse(predictions, targets)

        return loss_mse + 0.01 * loss_hash