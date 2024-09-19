import torch
import numpy as np
from pytorch3d.ops import sample_farthest_points
from common.pytorch_util import get_device
from typing import Union


def uniform_sampling(
    points: Union[torch.Tensor,  np.ndarray],
    num_points: int
) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(points, torch.Tensor):
        indices = torch.randperm(points.shape[-2])[:num_points]
    else:
        indices = np.random.permutation(points.shape[-2])[:num_points]
    return points[..., indices, :]


def farthest_point_sampling(
    points: Union[torch.Tensor,  np.ndarray],
    num_points: int,
    random_start: bool = False
) -> Union[torch.Tensor, np.ndarray]:
    unsqueezed = False
    if isinstance(points, torch.Tensor):
        device = points.device
        if points.ndim == 2:
            points = points[None]
            unsqueezed = True
        subsampled_pcd = sample_farthest_points(
            points[..., :3].to(get_device(use_gpu=True)),
            K=num_points, random_start_point=random_start
        )[0].to(device)
    else:
        if points.ndim == 2:
            points = points[np.newaxis]
            unsqueezed = True
        device = get_device(use_gpu=True)
        subsampled_pcd = sample_farthest_points(
            torch.from_numpy(points[..., :3]).to(device),
            K=num_points, random_start_point=random_start
        )[0].cpu().numpy()
    if unsqueezed:
        points = points.squeeze(0)
        subsampled_pcd = subsampled_pcd.squeeze(0)
    return subsampled_pcd


def pointcloud_subsampling(
    points: Union[torch.Tensor,  np.ndarray],
    num_points: int,
    method: str = 'fps'
) -> Union[torch.Tensor, np.ndarray]:
    # points: (*, N, d)
    # num_points: int
    # method: str

    if method == 'uniform':
        return uniform_sampling(points, num_points)
    elif method == 'fps':
        return farthest_point_sampling(points, num_points)
    else:
        raise ValueError(f"Unsupported method: {method}")
    