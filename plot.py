import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from model.pinn.pinn_net import PINN_FCN
from model.kr_net.my_distribution import NormalDistribution
from model.kr_net.kr_net import KR_net

SP = 1000.0

if False:
    # one peak
    def real_solution(input: torch.Tensor):
        return (-SP * (input ** 2).sum(-1, keepdims=True)).exp()


    def s(input: torch.Tensor):
        return (4 * SP - 4 * SP ** 2 * (input ** 2).sum(-1, keepdims=True)) * (
                -SP * (input ** 2).sum(-1, keepdims=True)).exp()

if True:
    # two peak
    def real_solution(input: torch.Tensor):
        return ((-SP * ((input - 0.5) ** 2).sum(-1, keepdims=True)).exp()
                + (-SP * ((input + 0.5) ** 2).sum(-1, keepdims=True)).exp())


    def s(input: torch.Tensor):
        return (4 * SP - 4 * SP ** 2 * ((input - 0.5) ** 2).sum(-1, keepdims=True)) * (
                -SP * ((input - 0.5) ** 2).sum(-1, keepdims=True)).exp() + (
                4 * SP - 4 * SP ** 2 * ((input + 0.5) ** 2).sum(-1, keepdims=True)) * (
                -SP * ((input + 0.5) ** 2).sum(-1, keepdims=True)).exp()


def plot(model,
         model_name, epoch, iterations, isSave):
    # test input
    x, y = torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256)
    input_x, input_y = torch.meshgrid(x, y, indexing='ij')
    input_x, input_y = input_x.requires_grad_().reshape(-1, 1), input_y.requires_grad_().reshape(-1, 1)
    input = torch.cat([input_x, input_y], dim=1)

    # test output
    output = model(input)
    u_appr = output.clone().detach().numpy()
    dy_dx = torch.autograd.grad(output, input_x, grad_outputs=torch.ones_like(output, device=output.device),
                                create_graph=True)[0]
    dy_dy = torch.autograd.grad(output, input_y, grad_outputs=torch.ones_like(output, device=output.device),
                                create_graph=True)[0]
    dy_dxx = torch.autograd.grad(dy_dx, input_x, grad_outputs=torch.ones_like(dy_dx, device=dy_dx.device),
                                 create_graph=True)[0]
    dy_dyy = torch.autograd.grad(dy_dy, input_y, grad_outputs=torch.ones_like(dy_dy, device=dy_dy.device),
                                 create_graph=True)[0]
    laplace_u = (dy_dxx + dy_dyy).detach().numpy()

    # real output
    u_real = real_solution(input).detach().numpy()
    source = s(input).detach().numpy()

    # error
    u_error = np.abs(u_appr - u_real)
    average_l2_error = np.sum(u_error ** 2) / (256 * 256)
    loss_pde = np.abs(laplace_u + source)

    # plot
    subplots_dict = {'u_real': [u_real, (0, 0)],
                     'u_appr': [u_appr, (0, 1)],
                     'u_error': [u_error, (0, 2)],
                     'source': [source, (1, 0)],
                     'laplace_u': [laplace_u, (1, 1)],
                     'loss_pde': [loss_pde, (1, 2)]}
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, layout="constrained")
    for key, value in subplots_dict.items():
        axs[value[1]].set_title(key)
        ax = axs[value[1]].scatter(input_x.reshape(-1).detach().numpy(), input_y.reshape(-1).detach().numpy(),
                                   c=value[0], s=10,
                                   cmap='rainbow')
        fig.colorbar(ax, ax=axs[value[1]])
    u_error[np.isnan(u_error)] = 0
    error_max = np.max(np.abs(u_error))
    plt.suptitle(f"model_epoch: {load_model_epoch} model_iterations: {load_model_iter} "
                 f"\n error_max: {error_max} \n average_l2_error: {average_l2_error}")
    plt.show()
    if isSave:
        dir_path = f'./plots/{model_name}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        fig.savefig(f'./plots/{model_name}/{epoch}_{iterations}.png')


def plot_samples(kr_net_model, reference_distribution,
                 model_name, epoch, iterations, isSave):
    z = reference_distribution.sample((2000, 2))
    S_domain_kr, _, _ = kr_net_model(z, reverse=True)
    S_domain_kr = S_domain_kr.detach()

    plt.scatter(S_domain_kr[:, 0:1].clone().reshape(-1).cpu().detach().numpy(),
                S_domain_kr[:, 1:2].clone().reshape(-1).cpu().detach().numpy(),
                s=1)
    # plt.show()
    if isSave:
        dir_path = f'./plots/{model_name}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.savefig(f'./plots/{model_name}/krsamples_{epoch}_{iterations}.png')
        plt.cla()


if __name__ == '__main__':
    # losd model
    model = PINN_FCN(2, 1)
    # kr_model = KR_net()
    # distribution = NormalDistribution()

    # plot temp
    load_model_epoch = 49
    load_model_iter = 1500
    model.load_state_dict(torch.load(f'./saved_models/pinn_{load_model_epoch}_{load_model_iter}.pt'))
    plot(model, f"temp", load_model_epoch, load_model_iter, False)

    # plot pinn error
    # for load_model_epoch in range(50):
    #     model.load_state_dict(torch.load(
    #         f'./saved_models/xxx/pinn_{load_model_epoch}_{load_model_iter}.pt'))
    #     plot(model, f"xxx", load_model_epoch, load_model_iter, True)

    # plot kr net samples
    # for load_model_epoch in range(22):
    #     kr_model.load_state_dict(torch.load(
    #         f'./saved_models/xxx/kr_net_{load_model_epoch}_{load_model_iter}.pt'))
    #     plot_samples(kr_model, distribution,
    #                  f"xxx", load_model_epoch, load_model_iter, True)
