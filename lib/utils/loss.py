import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy_loss_with_soft_target(pred, soft_target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))

class CrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, method='average', weight=1, temp=1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(CrossEntropy, self).__init__()
        self.weight = torch.tensor(weight).cuda()
        self.method = method
        self.temp = torch.tensor(temp).cuda()

    def forward(self, x, target):
        if self.method == 'boosting':
            x_weight = x/self.temp
            logprobs_weiight = F.log_softmax(x_weight, dim=-1)
            nll_loss_weight = -logprobs_weiight.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss_weight = nll_loss_weight.squeeze(1)
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        loss = self.weight * nll_loss
        weight_sum = self.weight.sum()
        if self.method == 'boosting':
            return nll_loss_weight, loss.sum()/weight_sum
        return loss.sum()/weight_sum

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, method='average', smoothing=0.1, weight=None, temp=1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.weight = weight.cuda()
        self.method = method
        self.temp = torch.tensor(temp).cuda()


    def forward(self, x, target):
        if self.method == 'boosting':
            x_weight = x/self.temp
            logprobs_weiight = F.log_softmax(x_weight, dim=-1)
            nll_loss_weight = -logprobs_weiight.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss_weight = nll_loss_weight.squeeze(1)

        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * self.weight * nll_loss + self.smoothing * self.weight * smooth_loss
        weight_sum = self.weight.sum()
        if self.method == 'boosting':
            return nll_loss_weight, loss.sum()/weight_sum
        return loss.sum()/weight_sum
