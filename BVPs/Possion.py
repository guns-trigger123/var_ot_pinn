import torch
from abc import ABC, abstractmethod


class Possion(ABC):
    @abstractmethod
    def real_solution(self, input: torch.Tensor):
        pass

    @abstractmethod
    def s(self, input: torch.Tensor):
        pass


class MultiPeakPossion(Possion):
    def __init__(self, n=16, SP=1000):
        self.n = n
        self.SP = SP

    def real_solution(self, input: torch.Tensor):
        return ((-self.SP * ((input - 0.5) ** 2).sum(-1, keepdims=True)).exp()
                + (-self.SP * ((input + 0.5) ** 2).sum(-1, keepdims=True)).exp())

    def s(self, input: torch.Tensor):
        return (4 * self.SP - 4 * self.SP ** 2 * ((input - 0.5) ** 2).sum(-1, keepdims=True)) * (
                -self.SP * ((input - 0.5) ** 2).sum(-1, keepdims=True)).exp() + (
                4 * self.SP - 4 * self.SP ** 2 * ((input + 0.5) ** 2).sum(-1, keepdims=True)) * (
                -self.SP * ((input + 0.5) ** 2).sum(-1, keepdims=True)).exp()


class TwoPeakPossion(Possion):
    def __init__(self, SP=1000):
        self.SP = SP

    def real_solution(self, input: torch.Tensor):
        return ((-self.SP * ((input - 0.5) ** 2).sum(-1, keepdims=True)).exp()
                + (-self.SP * ((input + 0.5) ** 2).sum(-1, keepdims=True)).exp())

    def s(self, input: torch.Tensor):
        return (4 * self.SP - 4 * self.SP ** 2 * ((input - 0.5) ** 2).sum(-1, keepdims=True)) * (
                -self.SP * ((input - 0.5) ** 2).sum(-1, keepdims=True)).exp() + (
                4 * self.SP - 4 * self.SP ** 2 * ((input + 0.5) ** 2).sum(-1, keepdims=True)) * (
                -self.SP * ((input + 0.5) ** 2).sum(-1, keepdims=True)).exp()


class OnePeakPossion(Possion):
    def __init__(self, SP=1000):
        self.SP = SP

    def real_solution(self, input: torch.Tensor):
        return (-self.SP * (input ** 2).sum(-1, keepdims=True)).exp()

    def s(self, input: torch.Tensor):
        return (4 * self.SP - 4 * self.SP ** 2 * (input ** 2).sum(-1, keepdims=True)) * (
                -self.SP * (input ** 2).sum(-1, keepdims=True)).exp()
