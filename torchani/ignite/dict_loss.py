from torch.nn.modules.loss import _Loss


class DictLosses(_Loss):

    def __init__(self, losses):
        super(DictLosses, self).__init__()
        self.losses = losses

    def forward(self, input, other):
        total = 0
        for i in self.losses:
            total += self.losses[i](input[i], other[i])
        return total
