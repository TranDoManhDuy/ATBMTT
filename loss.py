import torch
import torch.nn as nn
import implement_SHA1
class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.lambda_loss_true = 5.0
    
    def forward(self, predictions, targets):
        return self.mse(predictions, targets)