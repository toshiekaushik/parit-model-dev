import torch
import numpy as np
import torch.nn as nn

class TaskSchedulerPersonalizationV2(nn.Module):

    def __init__(self,
                 input_dim: int
                 ):
        super().__init__()
        self.linear_layer = nn.Linear(input_dim, 1)

    def foward(self, input: torch.Tensor) -> float:
        val = self.linear_layer(input)
        return val



    