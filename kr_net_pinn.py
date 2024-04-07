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

SP = 1000.0


# one peak
def real_solution(input: torch.Tensor):
    return (-SP * (input ** 2).sum(-1, keepdims=True)).exp()


def s(input: torch.Tensor):
    return (4 * SP - 4 * SP ** 2 * (input ** 2).sum(-1, keepdims=True)) * (
            -SP * (input ** 2).sum(-1, keepdims=True)).exp()


# two peak possion
# def real_solution(input: torch.Tensor):
#     return ((-SP * ((input - 0.5) ** 2).sum(-1, keepdims=True)).exp()
#             + (-SP * ((input + 0.5) ** 2).sum(-1, keepdims=True)).exp())
#
#
# def s(input: torch.Tensor):
#     return (4 * SP - 4 * SP ** 2 * ((input - 0.5) ** 2).sum(-1, keepdims=True)) * (
#             -SP * ((input - 0.5) ** 2).sum(-1, keepdims=True)).exp() + (
#             4 * SP - 4 * SP ** 2 * ((input + 0.5) ** 2).sum(-1, keepdims=True)) * (
#             -SP * ((input + 0.5) ** 2).sum(-1, keepdims=True)).exp()


def train_pinn(model, criterion, optimizer, step_schedule,
               x, y, S_domain, S_boundary, iterations, epoch):
    for iter in range(iterations):
        u = model(S_domain)
        dy_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u, device=u.device),
                                    create_graph=True)[0]
        dy_dy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u, device=u.device),
                                    create_graph=True)[0]
        dy_dxx = torch.autograd.grad(dy_dx, x, grad_outputs=torch.ones_like(dy_dx, device=dy_dx.device),
                                     create_graph=True)[0]
        dy_dyy = torch.autograd.grad(dy_dy, y, grad_outputs=torch.ones_like(dy_dy, device=dy_dy.device),
                                     create_graph=True)[0]
        res_domain = -dy_dxx - dy_dyy - s(S_domain).detach()
        res_boundary = model(S_boundary) - real_solution(S_boundary).detach()
        if (iter + 1) == iterations:
            out_res_domain = res_domain.clone().detach()

        loss_domain = criterion(res_domain, torch.zeros_like(res_domain, device=res_domain.device))
        loss_boundary = criterion(res_boundary, torch.zeros_like(res_boundary, device=res_boundary.device))
        loss = loss_domain + 5.0 * loss_boundary

        loss.backward()
        optimizer.step()
        step_schedule.step()
        model.zero_grad()
        S_domain.grad = None
        S_boundary.grad = None

        if (iter + 1) % 1500 == 0:
            print(f"pinn epoch: {epoch} iter: {iter} " +
                  f"loss: {loss} loss_domain: {loss_domain} loss_boundary: {loss_boundary}")
            save_path = os.path.join('./saved_models/', f'pinn_{epoch}_{iter + 1}.pt')
            torch.save(model.state_dict(), save_path)

    return out_res_domain


def h_delta(input: torch.Tensor, delta):
    with torch.no_grad():
        cutoff = torch.ones_like(input, dtype=input.dtype, device=input.device)
        zeros = torch.zeros_like(input, dtype=input.dtype, device=input.device)
        cutoff = torch.where((input < 0) & (input > -2 * delta), input / (2 * delta) + 1, cutoff)
        cutoff = torch.where((input > 1) & (input < 1 + 2 * delta), -input / (2 * delta) + 1, cutoff)
        cutoff = torch.where((input < -2 * delta) | (input > 1 + 2 * delta), zeros, cutoff)
        return cutoff.prod(-1)


def train_kr_net(model, criterion, optimizer, step_schedule, distribution,
                 S_domain, out_res_domain, p_before, beta, iterations, epoch):
    input_temp = S_domain.clone().detach()
    # x0 = input_temp[:, 0:1].requires_grad_()
    # x1 = input_temp[:, 1:2].requires_grad_()
    x0 = input_temp[5000:, 0:1].requires_grad_()
    x1 = input_temp[5000:, 1:2].requires_grad_()
    input = torch.cat([x0, x1], dim=1)
    for iter in range(iterations):
        y, pdj, sldj = model(input, reverse=False)
        p = distribution.prob(y) * pdj
        log_p = distribution.log_prob(y) + sldj

        # aas loss function
        # dp_dx0 = torch.autograd.grad(p, x0, grad_outputs=torch.ones_like(p, device=p.device),
        #                              create_graph=True)[0]
        # dp_dx1 = torch.autograd.grad(p, x1, grad_outputs=torch.ones_like(p, device=p.device),
        #                              create_graph=True)[0]
        # dp_norm2 = dp_dx0 ** 2 + dp_dx1 ** 2
        # # residual = - beta * dp_norm2 / (p_before+1e-6) + (out_res_domain ** 2) * p / (p_before+1e-6)
        # residual = - beta * dp_norm2 / (p_before[5000:,:]+1e-6) + (out_res_domain[5000:,:] ** 2) * p / (p_before[5000:,:]+1e-6)
        # loss = (-residual).mean()

        # das loss function
        # h_d = h_delta(input, 0.005)
        # out_res_domain = out_res_domain * h_d
        with torch.no_grad():
            p = distribution.prob(y) * pdj
            if not torch.all(p > 0):
                sys.exit("Error: p is not all postive.")
        # loss = - log_p * (out_res_domain ** 2) / (p+1e-6)
        loss = - log_p * (out_res_domain[5000:, :] ** 2) / (p + 1e-6)
        loss = loss.mean()

        # weighted likelihood  loss function
        # loss0 = - log_p * (out_res_domain**2)
        # loss0 = - log_p * out_res_domain[5000:,:]**2
        # loss = loss0.mean()

        loss.backward()
        optimizer.step()
        step_schedule.step()
        model.zero_grad()

        if (iter + 1) % 100 == 0:
            print(f"kr net epoch: {epoch} iter: {iter} " + f"loss: {loss}")

        if (iter + 1) % 1500 == 0:
            print(f"kr net epoch: {epoch} iter: {iter} " + f"loss: {loss}")
            save_path = os.path.join('./saved_models/', f'kr_net_{epoch}_{iter + 1}.pt')
            torch.save(model.state_dict(), save_path)


if __name__ == '__main__':

    # init model
    pinn_model = PINN_FCN(2, 1)
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
    pinn_model.train()
    kr_net_model.train()

    criterion = torch.nn.MSELoss()

    opt_pinn = optim.Adam(pinn_model.parameters(), lr=0.0001)
    opt_kr_net = optim.Adam(kr_net_model.parameters(), lr=0.0001)

    step_schedule_pinn = optim.lr_scheduler.StepLR(step_size=10000, gamma=0.95, optimizer=opt_pinn)
    step_schedule_kr_net = optim.lr_scheduler.StepLR(step_size=10000, gamma=0.95, optimizer=opt_kr_net)

    reference_distribution = NormalDistribution()

    # init sampling
    NUM_BOUNDARY = 8000
    NUM_DOMAIN = 10000
    # x = (torch.rand(size=(NUM_DOMAIN, 1)) * 1.9 - 0.95).to(device).requires_grad_()
    # y = (torch.rand(size=(NUM_DOMAIN, 1)) * 1.9 - 0.95).to(device).requires_grad_()
    x = (torch.rand(size=(NUM_DOMAIN, 1)) * 2 - 1).to(device).requires_grad_()
    y = (torch.rand(size=(NUM_DOMAIN, 1)) * 2 - 1).to(device).requires_grad_()
    S_domain = torch.cat([x, y], dim=1)
    S_boundary = torch.cat([torch.cat([torch.ones(NUM_BOUNDARY, 1), torch.rand(NUM_BOUNDARY, 1) * 2 - 1], 1),
                            torch.cat([-torch.ones(NUM_BOUNDARY, 1), torch.rand(NUM_BOUNDARY, 1) * 2 - 1], 1),
                            torch.cat([torch.rand(NUM_BOUNDARY, 1) * 2 - 1, torch.ones(NUM_BOUNDARY, 1)], 1),
                            torch.cat([torch.rand(NUM_BOUNDARY, 1) * 2 - 1, -torch.ones(NUM_BOUNDARY, 1)], 1)])
    S_boundary = S_boundary.to(device)

    EPOCHS = 50
    for epoch in range(EPOCHS):
        # train pinn model
        out_res_domain = train_pinn(pinn_model, criterion, opt_pinn, step_schedule_pinn,
                                    x, y, S_domain, S_boundary, 1500, epoch)
        opt_pinn.state = collections.defaultdict(dict)

        with torch.no_grad():
            if epoch == 0:
                p_before = torch.ones([S_domain.shape[0], 1], device=S_domain.device) / 4.0
            else:
                y_before, pdj_before, _ = kr_net_model(S_domain, reverse=False)
                p_before = reference_distribution.prob(y_before) * pdj_before
                p_before = p_before.detach()

        # kr_net_model = KR_net().to(device)
        # opt_kr_net = optim.Adam(kr_net_model.parameters(), lr=0.0005)
        # step_schedule_kr_net = optim.lr_scheduler.StepLR(step_size=1000, gamma=0.9, optimizer=opt_kr_net)

        # train kr_net model
        if epoch < EPOCHS - 1:
            train_kr_net(kr_net_model, criterion, opt_kr_net, step_schedule_kr_net, reference_distribution,
                         S_domain, out_res_domain, p_before, 5, 1500, epoch)
            opt_kr_net.state = collections.defaultdict(dict)

        # resample
        # kr_net sample
        # z = reference_distribution.sample((NUM_DOMAIN / 2, 2)).cuda()
        z = reference_distribution.sample((NUM_DOMAIN, 2)).cuda()
        S_domain_kr, _, _ = kr_net_model(z, reverse=True)
        S_domain_kr = S_domain_kr.detach()

        plt.scatter(S_domain_kr[:, 0:1].clone().reshape(-1).cpu().detach().numpy(),
                    S_domain_kr[:, 1:2].clone().reshape(-1).cpu().detach().numpy(),
                    s=1)
        plt.show()

        # uniform sample
        # S_domain_uniform = (torch.rand(size=(NUM_DOMAIN / 2, 2)) * 1.9 - 0.95).cuda()
        S_domain_uniform = (torch.rand(size=(NUM_DOMAIN / 2, 2)) * 2 - 1).cuda()

        x = torch.cat([S_domain_kr[:, 0:1], S_domain_uniform[:, 0:1]], dim=0).requires_grad_()
        y = torch.cat([S_domain_kr[:, 1:2], S_domain_uniform[:, 1:2]], dim=0).requires_grad_()
        # x = S_domain_kr[:, 0:1].requires_grad_()
        # y = S_domain_kr[:, 1:2].requires_grad_()

        S_domain = torch.cat([x, y], dim=1)
        S_boundary = torch.cat([torch.cat([torch.ones(NUM_BOUNDARY, 1), torch.rand(NUM_BOUNDARY, 1) * 2 - 1], 1),
                                torch.cat([-torch.ones(NUM_BOUNDARY, 1), torch.rand(NUM_BOUNDARY, 1) * 2 - 1], 1),
                                torch.cat([torch.rand(NUM_BOUNDARY, 1) * 2 - 1, torch.ones(NUM_BOUNDARY, 1)], 1),
                                torch.cat([torch.rand(NUM_BOUNDARY, 1) * 2 - 1, -torch.ones(NUM_BOUNDARY, 1)], 1)])
        S_boundary = S_boundary.to(device)
