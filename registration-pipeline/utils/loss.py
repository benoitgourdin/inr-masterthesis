import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.distance import cdist

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
from torch.autograd.functional import jacobian

from utils.pointnet2 import pointnet2_utils

import h5py


class MeanSquaredError(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()

    def forward(self, registration_pred, registration_gt, coords):
        return self.loss(registration_pred, registration_gt)


def square_distance(src, dst):
    b, n, _ = src.shape
    _, m, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(b, n, 1)
    dist += torch.sum(dst ** 2, -1).view(b, 1, m)
    return dist


class ChamferDistance(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, registration_pred, registration_gt, coords):
        # PointPWC implementation
        gt_pc = coords + registration_gt
        gen_pc = coords + registration_pred
        pc1 = gt_pc.permute(0, 2, 1)
        pc2 = gen_pc.permute(0, 2, 1)
        sqr_dist12 = square_distance(pc1, pc2)
        dist1, _ = torch.topk(sqr_dist12, 1, dim=-1, largest=False, sorted=False)
        dist2, _ = torch.topk(sqr_dist12, 1, dim=1, largest=False, sorted=False)
        dist1 = dist1.squeeze(2)
        dist2 = dist2.squeeze(1)
        return dist1.sum(dim=1).mean() + dist2.sum(dim=1).mean()

        # # Former implementation
        # # one direction
        # gen_points_kd_tree = KDTree(gen_pc)
        # one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_pc)
        # gt_to_gen_chamfer = np.mean(np.square(one_distances))
        # # other direction
        # gt_points_kd_tree = KDTree(gt_pc)
        # two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_pc)
        # gen_to_gt_chamfer = np.mean(np.square(two_distances))
        # return gt_to_gen_chamfer + gen_to_gt_chamfer


class EarthMoversDistance(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, registration_pred, registration_gt, coords):
        gt_pc = coords.squeeze().cpu().detach().numpy() + registration_gt.squeeze().cpu().detach().numpy()
        gen_pc = coords.squeeze().cpu().detach().numpy() + registration_pred.squeeze().cpu().detach().numpy()
        d = cdist(gt_pc[:1000], gen_pc[:1000])
        assignment = linear_sum_assignment(d)
        return d[assignment].mean()


class DiVRoC(Function):
    @staticmethod
    def forward(ctx, input, grid, shape):
        device = input.device
        dtype = input.dtype
        output = -jacobian(lambda x: (F.grid_sample(x, grid) - input).pow(2).mul(0.5).sum(),
                           torch.zeros(shape, dtype=dtype, device=device))
        ctx.save_for_backward(input, grid, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid, output = ctx.saved_tensors
        B, C = input.shape[:2]
        input_dims = input.shape[2:]
        output_dims = grad_output.shape[2:]
        y = jacobian(lambda x: F.grid_sample(grad_output.unsqueeze(2).view(B * C, 1, *output_dims), x).mean(),
                     grid.unsqueeze(1).repeat(1, C, *([1] * (len(input_dims) + 1)))
                     .view(B * C, *input_dims, len(input_dims))).view(B, C, *input_dims, len(input_dims))
        grad_grid = (input.numel() * input.unsqueeze(-1) * y).sum(1)
        grad_input = F.grid_sample(grad_output, grid)
        return grad_input, grad_grid, None


class DivrocLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.HuberLoss(reduction='sum', delta=1.0)

    def forward(self, registration_pred, registration_gt, coords):
        device = registration_pred.device
        dtype = registration_pred.dtype
        gt_pc = coords + registration_gt
        gen_pc = coords + registration_pred
        gt_grid = gt_pc.squeeze().unsqueeze(0).unsqueeze(2).unsqueeze(3)
        gt_input = torch.ones([1, 1, gt_grid.shape[1], 1, 1], dtype=dtype, device=device)
        pred_grid = gen_pc.squeeze().unsqueeze(0).unsqueeze(2).unsqueeze(3)
        pred_input = torch.ones([1, 1, pred_grid.shape[1], 1, 1], dtype=dtype, device=device)
        shape = [1, 1, 128, 128, 128]

        pred_grid = torch.tensor([[[[[0.0, 0.0, 0.0]]], [[[-0.9, -0.9, -0.9]]], [[[0.9, 0.9, 0.9]]]]])
        gt_grid = torch.tensor([[[[[0.5, 0.5, 0.5]]], [[[0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0]]]]])
        pred_input = torch.ones([1, 1, 3, 1, 1], dtype=dtype)
        gt_input = torch.ones([1, 1, 3, 1, 1], dtype=dtype)
        shape = [1, 1, 3, 3, 3]

        pred_rasterized = DiVRoC.apply(pred_input, pred_grid, shape)
        gt_rasterized = DiVRoC.apply(gt_input, gt_grid, shape)


        # Test visualization of rasterized PC
        value_to_check = pred_rasterized[0, 0, 0, 0, 0]
        mask = pred_rasterized != value_to_check
        count_non_equal = mask.sum().item()
        # print(128 * 128 * 128)
        # print(count_non_equal)
        indices = torch.nonzero(mask, as_tuple=True)
        if indices[0].numel() > 0:
            first_non_zero_index = tuple(index[0].item() for index in indices)
            first_non_zero_value = pred_rasterized[first_non_zero_index]
            # print(f"One value different from -0: {first_non_zero_value.item()}")
        else:
            print("All values are equal to -0.")
        # grid = pred_rasterized.squeeze().cpu().numpy()
        grid = pred_rasterized.squeeze().cpu().detach().numpy()
        value_to_check = grid[0][0][0]
        mask = grid != value_to_check
        # print(mask.sum().item())
        with h5py.File('/home/mil/gourdin/inr_3d_data/voxelgrid_pred.h5', 'w') as f:
            f.create_dataset('voxelgrid', data=grid)
        value_to_check = gt_rasterized[0, 0, 0, 0, 0]
        mask = gt_rasterized != value_to_check
        count_non_equal = mask.sum().item()
        # print(128 * 128 * 128)
        # print(count_non_equal)
        indices = torch.nonzero(mask, as_tuple=True)
        if indices[0].numel() > 0:
            first_non_zero_index = tuple(index[0].item() for index in indices)
            first_non_zero_value = gt_rasterized[first_non_zero_index]
            # print(f"One value different from -0: {first_non_zero_value.item()}")
        else:
            print("All values are equal to -0.")
        # grid = pred_rasterized.squeeze().cpu().numpy()
        grid = gt_rasterized.squeeze().cpu().detach().numpy()
        value_to_check = grid[0][0][0]
        mask = grid != value_to_check
        # print(mask.sum().item())
        with h5py.File('/home/mil/gourdin/inr_3d_data/voxelgrid_gt.h5', 'w') as f:
            f.create_dataset('voxelgrid', data=grid)


        return self.loss(pred_rasterized, gt_rasterized)


def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)
    return new_points


def curvature(pc):
    # pc: B 3 N
    pc = pc.permute(0, 2, 1)
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim = -1, largest=False, sorted=False) # B N 10 3
    grouped_pc = index_points_group(pc, kidx)
    pc_curvature = torch.sum(grouped_pc - pc.unsqueeze(2), dim = 2) / 9.0
    return pc_curvature # B N 3


def computeChamfer(pc1, pc2):
    '''
    pc1: B 3 N
    pc2: B 3 M
    '''
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    sqrdist12 = square_distance(pc1, pc2) # B N M
    #chamferDist
    dist1, _ = torch.topk(sqrdist12, 1, dim = -1, largest=False, sorted=False)
    dist2, _ = torch.topk(sqrdist12, 1, dim = 1, largest=False, sorted=False)
    dist1 = dist1.squeeze(2)
    dist2 = dist2.squeeze(1)
    return dist1, dist2


def curvatureWarp(pc, warped_pc):
    warped_pc = warped_pc.permute(0, 2, 1)
    pc = pc.permute(0, 2, 1)
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim = -1, largest=False, sorted=False) # B N 10 3
    grouped_pc = index_points_group(warped_pc, kidx)
    pc_curvature = torch.sum(grouped_pc - warped_pc.unsqueeze(2), dim = 2) / 9.0
    return pc_curvature # B N 3


def computeSmooth(pc1, pred_flow):
    '''
    pc1: B 3 N
    pred_flow: B 3 N
    '''
    pc1 = pc1.permute(0, 2, 1)
    pred_flow = pred_flow.permute(0, 2, 1)
    sqrdist = square_distance(pc1, pc1) # B N N
    _, kidx = torch.topk(sqrdist, 9, dim = -1, largest=False, sorted=False)
    grouped_flow = index_points_group(pred_flow, kidx) # B N 9 3
    diff_flow = torch.norm(grouped_flow - pred_flow.unsqueeze(2), dim = 3).sum(dim = 2) / 8.0
    return diff_flow


def interpolateCurvature(pc1, pc2, pc2_curvature):
    '''
    pc1: B 3 N
    pc2: B 3 M
    pc2_curvature: B 3 M
    '''
    B, _, N = pc1.shape
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    pc2_curvature = pc2_curvature
    sqrdist12 = square_distance(pc1, pc2) # B N M
    dist, knn_idx = torch.topk(sqrdist12, 5, dim = -1, largest=False, sorted=False)
    grouped_pc2_curvature = index_points_group(pc2_curvature, knn_idx) # B N 5 3
    norm = torch.sum(1.0 / (dist + 1e-8), dim = 2, keepdim = True)
    weight = (1.0 / (dist + 1e-8)) / norm
    inter_pc2_curvature = torch.sum(weight.view(B, N, 5, 1) * grouped_pc2_curvature, dim = 2)
    return inter_pc2_curvature


def multiScaleChamferSmoothCurvature(pc1, pc2, pred_flows):
    f_curvature = 0.3
    f_smoothness = 1.0
    f_chamfer = 1.0
    num_scale = len(pred_flows)
    alpha = [0.02, 0.04, 0.08, 0.16]
    chamfer_loss = torch.zeros(1).cuda()
    smoothness_loss = torch.zeros(1).cuda()
    curvature_loss = torch.zeros(1).cuda()
    for i in range(num_scale):
        cur_pc1 = pc1[i].unsqueeze(0).permute(0, 2, 1) # B 3 N
        cur_pc2 = pc2[i].unsqueeze(0).permute(0, 2, 1)
        cur_flow = pred_flows[i].unsqueeze(0).permute(0, 2, 1) # B 3 N
        cur_pc2_curvature = curvature(cur_pc2)
        cur_pc1_warp = cur_pc1 + cur_flow
        dist1, dist2 = computeChamfer(cur_pc1_warp, cur_pc2)
        moved_pc1_curvature = curvatureWarp(cur_pc1, cur_pc1_warp)
        chamferLoss = dist1.sum(dim = 1).mean() + dist2.sum(dim = 1).mean()
        smoothnessLoss = computeSmooth(cur_pc1, cur_flow).sum(dim = 1).mean()
        inter_pc2_curvature = interpolateCurvature(cur_pc1_warp, cur_pc2, cur_pc2_curvature)
        curvatureLoss = torch.sum((inter_pc2_curvature - moved_pc1_curvature) ** 2, dim = 2).sum(dim = 1).mean()
        chamfer_loss += alpha[i] * chamferLoss
        smoothness_loss += alpha[i] * smoothnessLoss
        curvature_loss += alpha[i] * curvatureLoss
    total_loss = f_chamfer * chamfer_loss + f_curvature * curvature_loss + f_smoothness * smoothness_loss
    return total_loss, chamfer_loss, curvature_loss, smoothness_loss


class PointPWC(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, registration_pred, registration_gt, coords):
        pc1 = coords.unsqueeze(0)
        pc2 = (coords + registration_gt)
        pred_flows = registration_pred
        total_loss, chamfer_loss, curvature_loss, smoothness_loss = multiScaleChamferSmoothCurvature(
            pc1, pc2, pred_flows)
        return total_loss


def create_loss(loss_function):
    # Supervised loss
    if loss_function == "mse":
        return MeanSquaredError()
    # Self-supervised loss
    elif loss_function == "divroc":
        return DivrocLoss()
    elif loss_function == "chamfer":
        return ChamferDistance()
    elif loss_function == "emd":
        return EarthMoversDistance()
    # PointPWC self-supervised loss
    elif loss_function == "pointpwc":
        return PointPWC()
