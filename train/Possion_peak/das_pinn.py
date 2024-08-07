import os
import sys
import torch
import collections
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from utils.utils import *
from model.pinn.pinn_net import PINN_FCN
from model.kr_net.kr_net import KR_net
from model.kr_net.my_distribution import NormalDistribution
from BVPs.Possion import TwoPeakPossion
from model.module import WeightedMSELoss


def train_pinn(model, criterion, optimizer, step_schedule, bvp,
               loss_history, S_domain, S_boundary, p_before, iterations, epoch):
    p_before_inv = 1.0 / (p_before + 1e-6)
    sum_p_before_inv = torch.sum(p_before_inv)
    len = p_before.shape[0]
    w = len * p_before_inv / sum_p_before_inv

    for iter in range(iterations):
        u = model(S_domain)
        res_domain = -laplace(u, S_domain) - bvp.s(S_domain).detach()
        res_boundary = model(S_boundary) - bvp.real_solution(S_boundary).detach()
        if (iter + 1) == iterations:
            out_res_domain = res_domain.clone().detach()

        loss_domain = criterion(res_domain, torch.zeros_like(res_domain, device=res_domain.device), w)
        loss_boundary = criterion(res_boundary, torch.zeros_like(res_boundary, device=res_boundary.device), 1)
        loss = loss_domain + 100.0 * loss_boundary

        # with torch.autograd.detect_anomaly():
        #     loss.backward()
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
            save_path = os.path.join('../../saved_models/', f'pinn_{epoch}_{iter + 1}.pt')
            torch.save(model.state_dict(), save_path)

    return out_res_domain


def train_kr_net(model, criterion, optimizer, step_schedule, distribution,
                 loss_history, S_domain, out_res_domain, p_before, beta, iterations, epoch):
    def h_delta(input: torch.Tensor, delta):
        # for the out_res_domain correction
        with torch.no_grad():
            cutoff = torch.ones_like(input, dtype=input.dtype, device=input.device)
            zeros = torch.zeros_like(input, dtype=input.dtype, device=input.device)
            cutoff = torch.where((input < -1) & (input > -1 - 2 * delta), (input + 1) / (2 * delta) + 1, cutoff)
            cutoff = torch.where((input > 1) & (input < 1 + 2 * delta), -(input - 1) / (2 * delta) + 1, cutoff)
            cutoff = torch.where((input < -1 - 2 * delta) | (input > 1 + 2 * delta), zeros, cutoff)
            return cutoff.prod(-1)

    h_d = h_delta(S_domain, 0.005)
    out_res_domain = out_res_domain * h_d

    p_before_inv = 1.0 / (p_before + 1e-6)
    sum_p_before_inv = torch.sum(p_before_inv)
    len = p_before.shape[0]
    w = len * p_before_inv / sum_p_before_inv

    for iter in range(iterations):
        y, pdj, sldj = model(S_domain, reverse=False)
        p = distribution.prob(y) * pdj
        log_p = distribution.log_prob(y) + sldj

        loss = - log_p * (out_res_domain ** 2) * w
        loss = loss.mean()

        loss.backward()
        optimizer.step()
        step_schedule.step()
        model.zero_grad()

        if (iter + 1) % 100 == 0:
            loss_history["total"].append(loss)

        if (iter + 1) % 100 == 0:
            print(f"kr net epoch: {epoch} iter: {iter} loss:{loss}")

        if (iter + 1) % iterations == 0:
            save_path = os.path.join('../../saved_models/', f'kr_net_{epoch}_{iter + 1}.pt')
            torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    # init model
    pinn_model = PINN_FCN(2, 1, "sin")
    kr_net_model = KR_net()

    # use cuda or cpu
    USE_CUDA = True
    if USE_CUDA:
        device = torch.device('cuda')
        pinn_model = pinn_model.to(device)
        kr_net_model = kr_net_model.to(device)
    else:
        device = torch.device('cpu')

    # init training preparation
    criterion_pinn = WeightedMSELoss()
    criterion_kr_net = None
    opt_pinn = optim.Adam(pinn_model.parameters(), lr=0.0001)
    opt_kr_net = optim.Adam(kr_net_model.parameters(), lr=0.0008)
    step_schedule_pinn = optim.lr_scheduler.StepLR(step_size=10000, gamma=0.95, optimizer=opt_pinn)
    step_schedule_kr_net = optim.lr_scheduler.StepLR(step_size=10000, gamma=0.95, optimizer=opt_kr_net)
    reference_distribution = NormalDistribution()

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
    loss_kr_net = {"total": []}

    EPOCHS = 50
    for epoch in range(EPOCHS):
        with torch.no_grad():
            if epoch == 0:
                p_before = torch.ones([S_domain.shape[0], 1], device=S_domain.device) / 4.0
            else:
                y_before, pdj_before, _ = kr_net_model(S_domain, reverse=False)
                p_before = reference_distribution.prob(y_before) * pdj_before
                p_before = p_before.detach()

        # train pinn model
        pinn_model.train()
        out_res_domain = train_pinn(pinn_model, criterion_pinn, opt_pinn, step_schedule_pinn, two_peak_possion,
                                    loss_pinn, S_domain, S_boundary, p_before, 500, epoch)
        # opt_pinn.state = collections.defaultdict(dict)

        # uni samples for kr net training
        pinn_model.eval()
        S_domain = (torch.rand(size=(NUM_DOMAIN, 2)) * 2 - 1).to(device).requires_grad_()
        p_before = torch.ones([S_domain.shape[0], 1], device=S_domain.device) / 4.0
        u = pinn_model(S_domain)
        res_domain = -laplace(u, S_domain) - two_peak_possion.s(S_domain).detach()
        out_res_domain = res_domain.clone().detach()
        pinn_model.zero_grad()

        # train kr_net model
        if epoch < EPOCHS - 1:
            kr_net_model.train()
            train_kr_net(kr_net_model, criterion_kr_net, opt_kr_net, step_schedule_kr_net, reference_distribution,
                         loss_kr_net, S_domain, out_res_domain, p_before, 5, 1000, epoch)
            # opt_kr_net.state = collections.defaultdict(dict)

        # kr_net resample
        kr_net_model.eval()
        z = reference_distribution.sample((NUM_DOMAIN, 2)).cuda()
        S_domain_kr, _, _ = kr_net_model(z, reverse=True)
        S_domain_pos = torch.ones_like(S_domain_kr, device=S_domain_kr.device)
        S_domain_neg = -torch.ones_like(S_domain_kr, device=S_domain_kr.device)
        S_domain_kr = torch.where((S_domain_kr > 1), S_domain_pos, S_domain_kr)
        S_domain_kr = torch.where((S_domain_kr < -1), S_domain_neg, S_domain_kr)

        plt.figure(figsize=(10, 10))
        plt.scatter(S_domain_kr[:, 0:1].clone().reshape(-1).cpu().detach().numpy(),
                    S_domain_kr[:, 1:2].clone().reshape(-1).cpu().detach().numpy(),
                    s=1)
        plt.xlim(-1.0, 1.0)
        plt.ylim(-1.0, 1.0)
        # plt.show()
        plt_path = f'../../plots/das_5000dom_1250bou/samples'
        plt.savefig(plt_path + f'/das_samples_{epoch}_500.png')
        plt.cla()

        S_domain = S_domain_kr.detach().requires_grad_()
        S_boundary = torch.cat([torch.cat([torch.ones(NUM_BOUNDARY, 1), torch.rand(NUM_BOUNDARY, 1) * 2 - 1], 1),
                                torch.cat([-torch.ones(NUM_BOUNDARY, 1), torch.rand(NUM_BOUNDARY, 1) * 2 - 1], 1),
                                torch.cat([torch.rand(NUM_BOUNDARY, 1) * 2 - 1, torch.ones(NUM_BOUNDARY, 1)], 1),
                                torch.cat([torch.rand(NUM_BOUNDARY, 1) * 2 - 1, -torch.ones(NUM_BOUNDARY, 1)], 1)])
        S_boundary = S_boundary.to(device)

    # save loss history
    loss_pinn_domain = torch.tensor(loss_pinn["domain"])
    loss_pinn_boundary = torch.tensor(loss_pinn["boundary"])
    loss_pinn_total = torch.tensor(loss_pinn["total"])
    loss_kr = torch.tensor(loss_kr_net["total"])
    torch.save(loss_pinn_domain, os.path.join('../../saved_models/', f'loss_pinn_domain.pt'))
    torch.save(loss_pinn_boundary, os.path.join('../../saved_models/', f'loss_pinn_boundary.pt'))
    torch.save(loss_pinn_total, os.path.join('../../saved_models/', f'loss_pinn_total.pt'))
    torch.save(loss_kr, os.path.join('../../saved_models/', f'loss_kr.pt'))
