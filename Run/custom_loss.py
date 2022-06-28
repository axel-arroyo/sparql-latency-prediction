import warnings
import torch
from torch import Tensor
from torch.nn import MSELoss


class CustomMSELoss(MSELoss):
    
    __constants__ = ['reduction']

    def __init__(self, threshold, weight_factor, size_average=None, reduce=None, reduction: str = 'mean',CUDA=False) -> None:
        super(CustomMSELoss, self).__init__(size_average, reduce, reduction)
        self.threshold = threshold
        self.weight_factor = weight_factor,
        self.CUDA = CUDA
    
    def mse_loss(self, input, target, size_average=None, reduce=None, reduction='mean'):
             
        if not (target.size() == input.size()):
            warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                          "This will likely lead to incorrect results due to broadcasting. "
                          "Please ensure they have the same size.".format(target.size(), input.size()),
                          stacklevel=2)

        expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        """Get vector of weights to apply, for each example if greater than  threshold apply factor weight_factor"""
        weights = torch.ones(expanded_input.size())
        if self.CUDA:
            weights = weights.cuda()
        expanded_input_threshold =expanded_input <= self.threshold
        weights = torch.where(expanded_input_threshold, weights, weights*self.weight_factor)
        return self.weighted_mse_loss(input, target, weights)

    def weighted_mse_loss(self, input, target, weight):
        return torch.mean(weight * ((input - target) ** 2))
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        
        loss = self.mse_loss(input, target, reduction=self.reduction)
        return loss
