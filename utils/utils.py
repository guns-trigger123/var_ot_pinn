import torch


def gradients(u, x, order=1):
    if order == 1:
        temp = torch.autograd.grad(u, x, torch.ones_like(u, device=u.device),
                                   create_graph=True, materialize_grads=True)[0]
        return temp.requires_grad_()
    else:
        return gradients(gradients(u, x), x, order=order - 1)


def logabs(x: torch.Tensor):
    return torch.log(torch.abs(x))


if __name__ == '__main__':
    def exp_reducer(x):
        return (x ** 2).sum(dim=1, keepdims=True)


    x = torch.arange(0, 6, dtype=torch.float32, requires_grad=True).reshape(3, 2)
    print("x", x)

    # y = torch.autograd.functional.jacobian(exp_reducer, x, create_graph=True)
    # print("y", y)

    if False:
        v = torch.ones([x.shape[0], 1])
        y_vjp0, y_vjp1 = torch.autograd.functional.vjp(exp_reducer, x, v, create_graph=True)
        print("y_vjp0", y_vjp0)
        print("y_vjp1", y_vjp1)

        z = y_vjp1.sum(-1, keepdims=True)
        print("z", z)

        criterion = torch.nn.MSELoss()
        loss = criterion(z, torch.zeros_like(z))
        print(loss)

        loss.backward()

    if True:
        y = exp_reducer(x)
        print("y", y)

        # dy_dx = torch.autograd.grad((y,), (x,), (torch.ones_like(y),),
        #                             create_graph=True)[0].requires_grad_()
        # print('dy_dx', dy_dx)
        # d2y_dx2 = torch.autograd.grad((dy_dx,), (x,), (torch.ones_like(dy_dx),),
        #                               create_graph=True)[0].requires_grad_()
        # print('d2y_dx2', d2y_dx2)
        # d3y_dx3 = torch.autograd.grad((d2y_dx2,), (x,), (torch.ones_like(d2y_dx2),),
        #                               create_graph=True)[0].requires_grad_()
        # print('d3y_dx3', d3y_dx3)
        # d4y_dx4 = torch.autograd.grad((d3y_dx3,), (x,), (torch.ones_like(d3y_dx3),),
        #                               create_graph=True)[0].requires_grad_()
        # print('d4y_dx4', d4y_dx4)

        dny_dxn = gradients(y, x, 5)
        print('dny_dxn', dny_dxn)

        z = dny_dxn.sum(-1, keepdims=True)
        print("z", z)

        criterion = torch.nn.MSELoss()
        loss = criterion(z, torch.zeros_like(z))
        print(loss)

        loss.backward()
        pass

    if False:
        def my_function(x, y):
            # x 并未直接影响最终结果
            z = y * 3
            w = z + 2
            return w


        x = torch.tensor([3.0], requires_grad=True)
        y = torch.tensor([5.0], requires_grad=True)

        result = my_function(x, y)

        # 计算关于 w 对 x 的梯度，但 x 对于最终结果并没有直接的影响
        grads = torch.autograd.grad(result, x, materialize_grads=True)
        print(grads[0])  # 输出 tensor([0.])
