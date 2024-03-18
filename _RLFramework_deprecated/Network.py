import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, *args, device=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = None
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def set_optimizer(self, optimizer):
        """
        Must be called before training.
        :param optimizer: An optimizer of network.
        """
        self.optimizer = optimizer

    def predict(self, x):
        """
        Gets predicted value based on network.
        It doesn't cause gradients. When needed gradients, use Network()
        :param x: Input data.
        :return: Output of network.
        """
        self.eval()

        with torch.no_grad():
            res = self(torch.FloatTensor(x).to(self.device))

        return res

    def train_batch(self, x, y, loss_function, clip: float = None):
        """
        :param x: Input data.
        :param y: Target data.
        :param loss_function: Loss function for (pred, y).
        :param clip: If not None, it clips gradient based on norm.
        :return: Loss of a train step.
        """
        self.train()
        self.optimizer.zero_grad()

        pred = self(torch.FloatTensor(x).to(self.device))
        loss = loss_function(pred, torch.FloatTensor(y).to(self.device))

        loss.backward()

        if clip is not None:
            nn.utils.clip_grad_norm_(self.parameters(), clip)

        self.optimizer.step()

        return loss.item()
