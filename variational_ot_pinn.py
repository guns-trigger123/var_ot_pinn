import os
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from utils.utils import *
from model.pinn.pinn_net import PINN_FCN
from model.ot.pyOMT_raw import *
from BVPs.Possion import TwoPeakPossion


def train_pinn(model, criterion, optimizer, step_schedule, bvp,
               loss_history, S_domain, S_boundary, iterations, epoch):
    for iter in range(iterations):
        u = model(S_domain)
        res_domain = -laplace(u, S_domain) - bvp.s(S_domain).detach()
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


def cal_nu(model, h_P, s_P):
    u_P = model(h_P)
    res_domain = (-laplace(u_P, h_P) - s_P).abs().detach().reshape(-1)
    res_domain /= torch.sum(res_domain)
    h_P.grad = None
    return res_domain


def cal_distance(C, d_volP):
    bat_size_n = d_volP.shape[0]
    num_C = C.shape[0]
    batch_size_C = num_C
    i = 0
    while i < num_C // batch_size_C:
        temp_C = C[i * batch_size_C:(i + 1) * batch_size_C]
        temp_C = temp_C.view(temp_C.shape[0], -1)  # useless?
        '''distance'''
        temp_C = torch.unsqueeze(temp_C, 1).repeat(1, bat_size_n, 1)
        d_volP = torch.unsqueeze(d_volP, 0).repeat(batch_size_C, 1, 1)
        distance = torch.sqrt(((temp_C - d_volP) ** 2).sum(-1))
        '''sort'''
        _, I = torch.sort(distance, dim=0, descending=False)
        '''add step'''
        i = i + 1
    return I


def gen_P_v3(p_s: pyOMT_raw, numX, thresh=1, dataset_name=None, k=1):
    if p_s.num_P != p_s.bat_size_P:
        sys.exit("Error: (num_p) is not equal to (batch_size_P), this method could generate points correctly.")
    if (numX % p_s.bat_size_n != 0) & (numX > p_s.bat_size_n):
        sys.exit('Error: (numX) is not a multiple of (p_s.bat_size_n)')
    # calculate mu-mass center
    p_s.pre_cal(0)
    p_s.cal_measure(True)
    C = p_s.d_c[~p_s.zero_location, :]
    P = p_s.h_P[~p_s.zero_location, :]
    # generate points
    S_all = torch.empty([numX, p_s.dim], dtype=torch.float, device=torch.device('cuda'))
    I_all = torch.empty([p_s.dim + k, numX], dtype=torch.long)  # +1
    is_generate = torch.ones(numX, dtype=torch.bool)
    if numX < p_s.bat_size_n:
        num_bat_x = 1
    else:
        num_bat_x = numX // p_s.bat_size_n
    for ii in range(num_bat_x):
        p_s.pre_cal(ii)
        # p_s.gen_pre_cal(ii)
        if numX < p_s.bat_size_n:
            S_all.copy_(p_s.d_volP[0:numX, :])
            I = cal_distance(C, S_all.clone())
            for iii in range(p_s.dim + k):  # +1
                I_all[iii, :].copy_(I[iii, :])
        else:
            S_all[ii * p_s.bat_size_n:(ii + 1) * p_s.bat_size_n, :].copy_(p_s.d_volP)
            I = cal_distance(C, p_s.d_volP)
            for iii in range(p_s.dim + k):  # +1
                I_all[iii, ii * p_s.bat_size_n:(ii + 1) * p_s.bat_size_n].copy_(I[iii, :])

    # nm are normal vector of the plane
    nm = torch.cat([P, -torch.ones(P.shape[0], 1, dtype=torch.float64)], dim=1)
    nm /= torch.norm(nm, dim=1).view(-1, 1)  # shape: (num_p, 1)
    for i_generate in range(p_s.dim + k - 1):  # +1
        cs = torch.sum(nm[I_all[0, :], :] * nm[I_all[i_generate + 1, :], :], 1)  # element-wise multiplication
        theta = torch.acos(cs)
        is_generate &= (theta < thresh)
    S_gen = S_all[is_generate, :]
    I_gen = I_all[:, is_generate]
    # P_gen = torch.zeros(size=S_gen.shape, dtype=torch.float)
    S_temp = torch.cat([S_gen, torch.ones([S_gen.shape[0], 1], dtype=torch.float, device=S_gen.device)], 1)
    S_vec = torch.unsqueeze(S_temp, 2)

    C_matrix = torch.ones([S_gen.shape[0], p_s.dim + 1, p_s.dim + 1], dtype=torch.float, device=torch.device('cuda'))
    P_matrix = torch.ones([S_gen.shape[0], p_s.dim + 1, p_s.dim + 1], dtype=torch.float, device=torch.device('cuda'))

    for i_t in range(p_s.dim + k):  # +1
        temp_c = C[I_gen[i_t, :], :]
        temp_p = P[I_gen[i_t, :], :]
        C_matrix[:, 0:2, i_t].copy_(temp_c)
        P_matrix[:, 0:2, i_t].copy_(temp_p)
    T = torch.bmm(P_matrix, torch.linalg.inv(C_matrix))
    P_gen = torch.bmm(T, S_vec)[:, 0:2, 0]
    P_gen = P_gen.cpu()

    # TODO: 求逆可能有误差
    out_range = P_gen.abs() <= 1
    out_range = out_range[:, 0] & out_range[:, 1]
    P_gen = P_gen[out_range]

    numGen = P_gen.shape[0]
    print('OT successfully generated {} samples'.format(numGen))

    # if p_s.dim == 2:
    #     fig2, ax2 = plt.subplots(1, 2, sharey=True, figsize=(14, 7))
    #     fig2.suptitle('Dataset ' + dataset_name, fontsize=16)
    #     ax2[0].scatter(P[:, 0], P[:, 1], marker='+', color='orange', label='Real')
    #     ax2[0].set_title('Groud-truth samples')
    #     ax2[0].axis('equal')
    #     ax2[1].scatter(P[:, 0], P[:, 1], marker='+', color='orange', label='Real')
    #     # ax2[1].scatter(P_gen[:, 0], P_gen[:, 1], marker='+', color='green', label='Generated', s=10, alpha=0.2)
    #     ax2[1].scatter(P_gen[:, 0], P_gen[:, 1], marker='+', color='green', label='Generated', s=10, alpha=0.8)
    #     ax2[1].set_title('Generated samples')
    #     ax2[1].axis('equal')
    #     plt.show()

    '''clear temporaty files'''
    clear_temp_data()

    return P_gen[:, 0:1].clone(), P_gen[:, 1:2].clone()


def gen_P(p_s: pyOMT_raw, numX, thresh=1, dataset_name=None, k=1):
    if p_s.num_P != p_s.bat_size_P:
        sys.exit("Error: (num_p) is not equal to (batch_size_P), this method could generate points correctly.")
    if (numX % p_s.bat_size_n != 0) & (numX > p_s.bat_size_n):
        sys.exit('Error: (numX) is not a multiple of (p_s.bat_size_n)')
    # calculate mu-mass center
    p_s.pre_cal(0)
    p_s.cal_measure(True)
    C = p_s.d_c[~p_s.zero_location, :]
    P = p_s.h_P[~p_s.zero_location, :]
    # generate points
    S_all = torch.empty([numX, p_s.dim], dtype=torch.float, device=torch.device('cuda'))
    I_all = torch.empty([p_s.dim + k, numX], dtype=torch.long)  # +1
    is_generate = torch.ones(numX, dtype=torch.bool)
    if numX < p_s.bat_size_n:
        num_bat_x = 1
    else:
        num_bat_x = numX // p_s.bat_size_n
    for ii in range(num_bat_x):
        p_s.pre_cal(ii)
        # p_s.gen_pre_cal(ii)
        if numX < p_s.bat_size_n:
            S_all.copy_(p_s.d_volP[0:numX, :])
            I = cal_distance(C, S_all.clone())
            for iii in range(p_s.dim + k):  # +1
                I_all[iii, :].copy_(I[iii, :])
        else:
            S_all[ii * p_s.bat_size_n:(ii + 1) * p_s.bat_size_n, :].copy_(p_s.d_volP)
            I = cal_distance(C, p_s.d_volP)
            for iii in range(p_s.dim + k):  # +1
                I_all[iii, ii * p_s.bat_size_n:(ii + 1) * p_s.bat_size_n].copy_(I[iii, :])

    # nm are normal vector of the plane
    nm = torch.cat([P, -torch.ones(P.shape[0], 1, dtype=torch.float64)], dim=1)
    nm /= torch.norm(nm, dim=1).view(-1, 1)  # shape: (num_p, 1)
    for i_generate in range(p_s.dim + k - 1):  # +1
        cs = torch.sum(nm[I_all[0, :], :] * nm[I_all[i_generate + 1, :], :], 1)  # element-wise multiplication
        theta = torch.acos(cs)
        is_generate &= (theta < thresh)
    S_gen = S_all[is_generate, :]
    I_gen = I_all[:, is_generate]
    P_gen = torch.zeros(size=S_gen.shape, dtype=torch.float)
    lbd_gen = torch.empty(size=I_gen.shape, dtype=torch.float)

    numGen = I_gen.shape[1]
    print('ot successfully generated {} samples'.format(numGen))

    for i_lbd in range(p_s.dim + k):  # +1
        temp_c = C[I_gen[i_lbd, :], :]
        temp_lbd = 1.0 / torch.norm(S_gen - temp_c, dim=1, keepdim=True)
        lbd_gen[i_lbd:i_lbd + 1, :].copy_(temp_lbd.t())
    lbd_gen /= torch.sum(lbd_gen, dim=0, keepdim=True)
    for i_p in range(p_s.dim + k):  # +1
        P_gen += torch.mul(P[I_gen[i_p, :], :], lbd_gen[i_p:i_p + 1, :].t())

    if p_s.dim == 2:
        fig2, ax2 = plt.subplots(1, 2, sharey=True, figsize=(14, 7))
        fig2.suptitle('Dataset ' + dataset_name, fontsize=16)
        ax2[0].scatter(P[:, 0], P[:, 1], marker='+', color='orange', label='Real')
        ax2[0].set_title('Groud-truth samples')
        ax2[0].axis('equal')
        ax2[1].scatter(P[:, 0], P[:, 1], marker='+', color='orange', label='Real')
        ax2[1].scatter(P_gen[:, 0], P_gen[:, 1], marker='+', color='green', label='Generated', s=10, alpha=0.2)
        ax2[1].set_title('Generated samples')
        ax2[1].axis('equal')
        plt.show()

    '''clear temporaty files'''
    clear_temp_data()

    return P_gen[:, 0:1].clone(), P_gen[:, 1:2].clone()


if __name__ == '__main__':
    # init model
    pinn_model = PINN_FCN(2, 1, "sin")

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
    NUM_DOMAIN = 1000
    NUM_BOUNDARY = NUM_DOMAIN // 4
    S_domain = (torch.rand(size=(NUM_DOMAIN, 2)) * 2 - 1).to(device).requires_grad_()
    S_boundary = torch.cat([torch.cat([torch.ones(NUM_BOUNDARY, 1), torch.rand(NUM_BOUNDARY, 1) * 2 - 1], 1),
                            torch.cat([-torch.ones(NUM_BOUNDARY, 1), torch.rand(NUM_BOUNDARY, 1) * 2 - 1], 1),
                            torch.cat([torch.rand(NUM_BOUNDARY, 1) * 2 - 1, torch.ones(NUM_BOUNDARY, 1)], 1),
                            torch.cat([torch.rand(NUM_BOUNDARY, 1) * 2 - 1, -torch.ones(NUM_BOUNDARY, 1)], 1)])
    S_boundary = S_boundary.to(device)

    # init BVP
    two_peak_possion = TwoPeakPossion()

    # init h_P and nu_mass_P
    x_P, y_P = torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32)
    x_P, y_P = torch.meshgrid(x_P, y_P, indexing='ij')
    x_P, y_P = x_P.reshape(-1, 1), y_P.reshape(-1, 1)
    h_P = torch.cat([x_P, y_P], dim=1).to(device).requires_grad_()
    h_P_detach = h_P.clone().detach().cpu()
    num_P = h_P.shape[0]
    nu_mass_P = torch.empty(num_P, dtype=h_P.dtype)
    s_P = two_peak_possion.s(h_P).detach()

    # loss track
    loss_pinn = {"domain": [], "boundary": [], "total": []}

    EPOCHS = 50
    for epoch in range(EPOCHS):
        # train pinn model
        pinn_model.train()
        train_pinn(pinn_model, criterion, opt_pinn, step_schedule_pinn, two_peak_possion,
                   loss_pinn, S_domain, S_boundary, 500, epoch)

        pinn_model.eval()
        # cal nu_mass_P
        nu_mass_P.copy_(cal_nu(pinn_model, h_P, s_P))

        # train varitional ot
        p_s = pyOMT_raw(h_P_detach, num_P, 2, 10000, 5e-4, num_P, 10000, nu_mass_P)
        train_omt(p_s, 1)

        # ot resample
        x, y = gen_P_v3(p_s, NUM_DOMAIN, 0.5, "grid", 1)

        S_domain_ot = torch.cat([x, y], dim=1)
        plt.figure(figsize=(10, 10))
        plt.scatter(S_domain_ot[:, 0:1].clone().reshape(-1).cpu().detach().numpy(),
                    S_domain_ot[:, 1:2].clone().reshape(-1).cpu().detach().numpy(),
                    s=1)
        plt.xlim(-1.0, 1.0)
        plt.ylim(-1.0, 1.0)
        # plt.show()
        plt_path = f'./plots/ot_1000dom_250bou/samples'
        plt.savefig(plt_path + f'/ot_samples_{epoch}_500.png')
        plt.cla()

        S_domain = S_domain_ot.detach().to(device).requires_grad_()
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
