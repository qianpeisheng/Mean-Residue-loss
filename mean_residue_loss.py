
from torch import nn
import math
import torch
import numpy as np
import torch.nn.functional as F

class MeanResidueLoss(nn.Module):

    def __init__(self, lambda_1, lambda_2, start, end, K=3):
        super().__init__()
        np.random.seed(2019)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.start = start
        self.end = end
        self.K = K

    def forward(self, input, target):

        N = input.size()[0]
        target = target.type(torch.FloatTensor).cuda()
        m = nn.Softmax(dim=1)
        p = m(input)

        # mean loss
        a = torch.arange(self.start, self.end + 1, dtype=torch.float32).cuda()
        mean = torch.squeeze((p * a).sum(1, keepdim=True), dim=1)
        mse = (mean - target)**2
        mean_loss = mse.mean() / 2.0

        # resdue loss
        no_top_k = self.end -self.start + 1 - self.K
        EPS = 1e-3

        # print(no_top_k)
        p_not_K, _ = torch.topk(p, no_top_k, dim =1, largest=False) # _ means indexs

        residue_loss = (-(p_not_K + EPS) * torch.log(p_not_K + EPS)).sum(1,keepdim = True).mean()

        return self.lambda_1 * mean_loss, self.lambda_2 * residue_loss, -1
