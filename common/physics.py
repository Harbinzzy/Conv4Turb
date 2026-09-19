import torch
import torch.nn.functional as F


def _gradient(volume, spacing, dim):
    if volume.ndim != 5:
        raise ValueError(f"Expected a 5D tensor, got shape {tuple(volume.shape)}")
    spacing = spacing.reshape(-1)
    gradients = [
        torch.gradient(
            volume[index : index + 1],
            spacing=spacing[index],
            dim=dim,
            edge_order=1,
        )[0]
        for index in range(volume.shape[0])
    ]
    return torch.cat(gradients, dim=0)


def torch_dx(volume, spacing):
    return _gradient(volume, spacing, dim=2)


def torch_dy(volume, spacing):
    return _gradient(volume, spacing, dim=3)


def torch_dz(volume, spacing):
    return _gradient(volume, spacing, dim=4)


def remove_edges(volume):
    return volume[:, :, 1:-1, 1:-1, 1:-1]


def box_filter(volume, filter_width):
    kernel = volume.new_ones((1, 1, filter_width, filter_width, filter_width))
    kernel /= filter_width**3
    return F.conv3d(volume, kernel, stride=filter_width)


def favre_filter(rho, field, filter_width):
    return box_filter(rho * field, filter_width) / box_filter(rho, filter_width)


def sgs(fields, filter_width):
    if fields.shape[1] != 4:
        raise ValueError(f"Expected rho/u/v/w channels, got {fields.shape[1]}")
    rho, u, v, w = fields[:, 0:1], fields[:, 1:2], fields[:, 2:3], fields[:, 3:4]
    rho_filtered = box_filter(rho, filter_width)
    u_filtered = favre_filter(rho, u, filter_width)
    v_filtered = favre_filter(rho, v, filter_width)
    w_filtered = favre_filter(rho, w, filter_width)

    stresses = torch.cat(
        (
            favre_filter(rho, u * u, filter_width) - u_filtered * u_filtered,
            favre_filter(rho, v * v, filter_width) - v_filtered * v_filtered,
            favre_filter(rho, w * w, filter_width) - w_filtered * w_filtered,
            favre_filter(rho, u * v, filter_width) - u_filtered * v_filtered,
            favre_filter(rho, u * w, filter_width) - u_filtered * w_filtered,
            favre_filter(rho, v * w, filter_width) - v_filtered * w_filtered,
        ),
        dim=1,
    )
    return stresses * rho_filtered


def divergence_sgs(stress, dx, dy, dz):
    tau11, tau22, tau33 = stress[:, 0:1], stress[:, 1:2], stress[:, 2:3]
    tau12, tau13, tau23 = stress[:, 3:4], stress[:, 4:5], stress[:, 5:6]
    div1 = torch_dx(tau11, dx) + torch_dy(tau12, dy) + torch_dz(tau13, dz)
    div2 = torch_dx(tau12, dx) + torch_dy(tau22, dy) + torch_dz(tau23, dz)
    div3 = torch_dx(tau13, dx) + torch_dy(tau23, dy) + torch_dz(tau33, dz)
    return div1, div2, div3


def nrmse(prediction, target, eps=1e-12):
    return torch.linalg.vector_norm(prediction - target) / (
        torch.linalg.vector_norm(target) + eps
    )
