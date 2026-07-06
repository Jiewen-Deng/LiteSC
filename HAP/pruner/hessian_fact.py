"""
@Author: JW Deng
@Time: 2026/7/6 17:00
"""
import torch
from progress.bar import Bar
import time

def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for name, param in model.named_parameters():
        # 不对嵌入层进行剪枝，因为嵌入层的处理方式与其他层不同
        if not param.requires_grad:
        # if not param.requires_grad or 'embedding' in name.lower():
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads


def hessian_vector_product(gradsH, params, v, stop_criterion=False):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(gradsH, params, grad_outputs=v, only_inputs=True, retain_graph = not stop_criterion)
    return hv

def get_trace_hut(model, data, pad_idx, criterion, device, n_v, loader, channel = 'AWGN', cuda = True, channelwise = False, layerwise = False):
    """
    compute the trace of hessian using Hutchinson's method
    """
    assert not (channelwise and layerwise)
    if loader:
        inputs = next(iter(data))
        targets = inputs
        # inputs, targets = next(iter(data))
    else:
        inputs = data
        targets = data
        # inputs, targets = data

    if cuda:
        inputs, targets = inputs.to(device), targets.to(device)

    else:
        device = 'cpu'
    model.eval()

    outputs, loss = model(inputs, 0.1, pad_idx, criterion, channel)
    loss.backward()# create_graph = True

    params, gradsH = get_params_grad(model)
    if channelwise:
        trace_vhv = [[[] for c in range(p.size(0))] for p in params]
    elif layerwise:
        trace_vhv = [[] for p in params]
    else:
        trace_vhv = []

    bar = Bar('Computing trace', max=n_v)
    for i in range(n_v):
        start_time = time.time()
        bar.suffix = f'({i + 1}/{n_v}) |ETA: {bar.elapsed_td}<{bar.eta_td}'
        bar.next()
        v = [torch.randint_like(p, high = 2, device = device).float() * 2 - 1 for p in params]
        if loader:
            THv = [torch.zeros(p.size()).to(device) for p in params]
            for inputs in data:
                inputs = inputs.to(device)
                model.zero_grad()
                outputs, loss = model(inputs, 0.1, pad_idx, criterion, 'AWGN')
                # outputs = model(inputs)
                # loss = criterion(outputs, targets)
                loss.backward(create_graph = True)

                params, gradsH = get_params_grad(model)
                Hv = torch.autograd.grad(gradsH, params, grad_outputs = v, only_inputs = True, retain_graph = False)
                # remenber to normalize over dummy-batch
                THv = [THv1 + Hv1/float(len(data)) + 0. for THv1, Hv1 in zip(THv, Hv)]
            Hv = THv
        else:
            Hv = hessian_vector_product(gradsH, params, v, stop_criterion= (i==(n_v-1)))
        Hv = [Hvi.detach().cpu() for Hvi in Hv]
        v = [vi.detach().cpu() for vi in v]
        with torch.no_grad():
            if channelwise:
                for Hv_i in range(len(Hv)):
                    for channel_i in range(Hv[Hv_i].size(0)):
                        trace_vhv[Hv_i][channel_i].append(Hv[Hv_i][channel_i].flatten().dot(v[Hv_i][channel_i].flatten()).item())
            elif layerwise:
                for Hv_i in range(len(Hv)):
                    trace_vhv[Hv_i].append(Hv[Hv_i].flatten().dot(v[Hv_i].flatten()).item())
            else:
                trace_vhv.append(group_product(Hv, v).item())
    bar.finish()
    return trace_vhv

