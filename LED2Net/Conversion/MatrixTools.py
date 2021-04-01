import torch
import copy
import pytorch3d.transforms.rotation_conversions as p3dr

__all__ = [
        'homogeneous',
        'angle_axis_to_rotation_matrix',
        'rotation_matrix_to_angle_axis',
        'pose_vector_to_projection_matrix'
    ]

def homogeneous(tensor: torch.Tensor):
    shape = list(copy.deepcopy(tensor.shape))
    shape[-1] = 1
    ones = torch.ones(*shape, dtype=tensor.dtype).to(tensor.device)
    t = torch.cat([tensor, ones], dim=-1)

    return t

def angle_axis_to_rotation_matrix(angle_axis: torch.Tensor): 
    R = p3dr.axis_angle_to_matrix(angle_axis)

    return R

def rotation_matrix_to_angle_axis(rotation_matrix: torch.Tensor):
    angle_axis = p3dr.matrix_to_axis_angle(rotation_matrix)

    return angle_axis

def pose_vector_to_projection_matrix(pose_vec, rotation='axis-angle'):
    R = angle_axis_to_rotation_matrix(pose_vec[..., :3])
    t = pose_vec[..., 3:].unsqueeze(-1)
    if rotation == 'axis-angle':
        Rt = torch.cat([R, t], dim=-1)
    else:
        raise NotImplementedError

    return Rt