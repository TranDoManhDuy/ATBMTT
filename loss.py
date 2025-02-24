import torch
import torch.nn as nn

class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.lambda_loss_true = 5.0
    
    def forward(self, predictions, targers):
        # predictions và targets có dạng (batch_size, 1)
        predictions_true = predictions * targers
        predictions_false = (1 - targers) * predictions
        
        loss_pred_true = torch.abs(torch.sum(predictions_true))
        loss_pred_false = torch.abs(torch.sum(predictions_true))
        
        return loss_pred_false + loss_pred_true * self.lambda_loss_true