import os
import sys
import torch
import collections
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from utils.utils import *
from model.pinn.pinn_net import PINN_FCN
from BVPs.Possion import TwoPeakPossion

def train_pinn(model, criterion, optimizer, step_schedule, bvp,
               loss_history, S_domain, S_boundary, iterations, epoch):
    for iter in range(iterations):
        u = model(S_domain)
        res_domain = -laplace(u,S_domain) - bvp.s(S_domain).detach()
        res_boundary = model(S_boundary) - bvp.real_solution(S_boundary).detach()

        loss_domain = criterion(res_domain, torch.zeros_like(res_domain, device=res_domain.device))
        loss_boundary = criterion(res_boundary, torch.zeros_like(res_boundary, device=res_boundary.device))
        loss = loss_domain + 100.0 * loss_boundary

        loss.backward()
        optimizer.step()
        step_schedule.step()
        model.zero_grad()
        S_domain.grad = None
        S_boundary.grad = None

        if (iter + 1) % 100 == 0:
            loss_history["domain"].append(loss_domain)
            loss_history["boundary"].append(loss_boundary)
            loss_history["total"].append(loss)

        if (iter + 1) % iterations == 0:
            print(f"pinn epoch: {epoch} iter: {iter} " +
                  f"loss: {loss} loss_domain: {loss_domain} loss_boundary: {loss_boundary}")

        if (iter + 1) % iterations == 0:
            save_path = os.path.join('./saved_models/', f'pinn_{epoch}_{iter + 1}.pt')
            torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    # init model
    pinn_model = PINN_FCN(2, 1,"sin")

    # use cuda or cpu
    USE_CUDA = True
    if USE_CUDA:
        device = torch.device('cuda')
        pinn_model = pinn_model.to(device)
    else:
        device = torch.device('cpu')

    # init training preparation
    criterion = torch.nn.MSELoss()
    opt_pinn = optim.Adam(pinn_model.parameters(), lr=0.0001)
    step_schedule_pinn = optim.lr_scheduler.StepLR(step_size=10000, gamma=0.95, optimizer=opt_pinn)

    # init sampling
    NUM_DOMAIN = 5000
    NUM_BOUNDARY = NUM_DOMAIN // 4
    S_domain = (torch.rand(size=(NUM_DOMAIN, 2)) * 2 - 1).to(device).requires_grad_()
    S_boundary = torch.cat([torch.cat([torch.ones(NUM_BOUNDARY, 1), torch.rand(NUM_BOUNDARY, 1) * 2 - 1], 1),
                            torch.cat([-torch.ones(NUM_BOUNDARY, 1), torch.rand(NUM_BOUNDARY, 1) * 2 - 1], 1),
                            torch.cat([torch.rand(NUM_BOUNDARY, 1) * 2 - 1, torch.ones(NUM_BOUNDARY, 1)], 1),
                            torch.cat([torch.rand(NUM_BOUNDARY, 1) * 2 - 1, -torch.ones(NUM_BOUNDARY, 1)], 1)])
    S_boundary = S_boundary.to(device)

    # init BVP
    two_peak_possion = TwoPeakPossion()

    # loss track
    loss_pinn = {"domain": [], "boundary": [], "total": []}

    EPOCHS = 50
    for epoch in range(EPOCHS):
        # train pinn model
        pinn_model.train()
        train_pinn(pinn_model, criterion, opt_pinn, step_schedule_pinn, two_peak_possion,
                   loss_pinn, S_domain, S_boundary, 500, epoch)

        # uniform resample
        S_domain = (torch.rand(size=(NUM_DOMAIN, 2)) * 2 - 1).to(device).requires_grad_()
        S_boundary = torch.cat([torch.cat([torch.ones(NUM_BOUNDARY, 1), torch.rand(NUM_BOUNDARY, 1) * 2 - 1], 1),
                                torch.cat([-torch.ones(NUM_BOUNDARY, 1), torch.rand(NUM_BOUNDARY, 1) * 2 - 1], 1),
                                torch.cat([torch.rand(NUM_BOUNDARY, 1) * 2 - 1, torch.ones(NUM_BOUNDARY, 1)], 1),
                                torch.cat([torch.rand(NUM_BOUNDARY, 1) * 2 - 1, -torch.ones(NUM_BOUNDARY, 1)], 1)])
        S_boundary = S_boundary.to(device)

    # save loss history
    loss_pinn_domain = torch.tensor(loss_pinn["domain"])
    loss_pinn_boundary = torch.tensor(loss_pinn["boundary"])
    loss_pinn_total = torch.tensor(loss_pinn["total"])
    torch.save(loss_pinn_domain, os.path.join('./saved_models/', f'loss_pinn_domain.pt'))
    torch.save(loss_pinn_boundary, os.path.join('./saved_models/', f'loss_pinn_boundary.pt'))
    torch.save(loss_pinn_total, os.path.join('./saved_models/', f'loss_pinn_total.pt'))