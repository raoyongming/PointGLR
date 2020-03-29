import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))
import torch
import torch.nn as nn
from torch.autograd import Variable
import pytorch_utils as pt_utils
from pointnet2_modules import PointnetSAModule, PointnetSAModuleMSGPN2
from pointnet2_utils import three_nn, three_interpolate
import numpy as np


class MetricLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss().cuda()

    def get_metric_loss(self, x, ref):
        '''
        :param x: (bs, n_rkhs)
        :param ref: (bs, n_rkhs, n_loc)
        :return: loss
        '''

        bs, n_rkhs, n_loc = ref.size()
        ref = ref.transpose(0, 1).reshape(n_rkhs, -1)
        score = torch.matmul(x, ref) * 64.  # (bs * n_loc, bs)
        score = score.view(bs, bs, n_loc).transpose(1, 2).reshape(bs * n_loc, bs)
        gt_label = torch.arange(bs, dtype=torch.long, device=x.device).view(bs, 1).expand(bs, n_loc).reshape(-1)
        return self.ce(score, gt_label)

    def forward(self, x, refs):
        loss = 0.
        for ref in refs:
            loss += self.get_metric_loss(x, ref)
        return loss


class ChamferLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        '''
        :param x: (bs, np, 3)
        :param y: (bs, np, 3)
        :return: loss
        '''

        x = x.unsqueeze(1)
        y = y.unsqueeze(2)
        dist = torch.sqrt(1e-6 + torch.sum(torch.pow(x - y, 2), 3)) # bs, ny, nx
        min1, _ = torch.min(dist, 1)
        min2, _ = torch.min(dist, 2)

        return min1.mean() + min2.mean()

def NormalLoss(x, y):
    loss = torch.sum(torch.abs(x * y), dim=1).mean()
    return 1 - loss

class Normalize(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=self.dim, keepdim=True)
        return x / norm

class PointNet2(nn.Module):

    def __init__(self, n_rkhs, input_channels=0, use_xyz=True, point_wise_out=False, multi=1.0):
        super().__init__()

        print('Using', multi, 'times PointNet SSG model')

        self.SA_modules = nn.ModuleList()

        self.point_wise_out = point_wise_out

        self.SA_modules.append(
            PointnetSAModuleMSGPN2(
                npoint=512,
                radii=[0.23],
                nsamples=[48],
                mlps=[[input_channels+3, int(multi * 64), int(multi * 128)]],
                use_xyz=use_xyz,
            )
        )

        self.SA_modules.append(
            PointnetSAModuleMSGPN2(
                npoint=128,
                radii=[0.32],
                nsamples=[64],
                mlps=[[int(multi * 128)+3, int(multi * 128), int(multi * 512)]],
                use_xyz=use_xyz,
            )
        )
        
        self.SA_modules.append(
            PointnetSAModule(
                nsample = 128,
                mlp=[int(multi * 512), int(multi * 1024)],
                use_xyz=use_xyz
            )
        )

        self.prediction_modules = nn.ModuleList()

        mid_channel = min(int(multi * 128), n_rkhs)
        self.prediction_modules.append(
            nn.Sequential(
                nn.Conv1d(int(multi * 128), mid_channel, 1),
                nn.BatchNorm1d(mid_channel),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_channel, n_rkhs, 1),
                Normalize(dim=1)
            )
        )

        mid_channel = min(int(multi * 512), n_rkhs)
        self.prediction_modules.append(
            nn.Sequential(
                nn.Conv1d(int(multi * 512), mid_channel, 1),
                nn.BatchNorm1d(mid_channel),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_channel, n_rkhs, 1),
                Normalize(dim=1)
            )
        )

        mid_channel = min(int(multi * 1024), n_rkhs)
        self.prediction_modules.append(
            nn.Sequential(
                nn.Conv1d(int(multi * 1024), mid_channel, 1),
                nn.BatchNorm1d(mid_channel),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_channel, n_rkhs, 1),
                Normalize(dim=1)
            )
        )

        self.adaptive_maxpool = nn.AdaptiveMaxPool1d(1)

        if point_wise_out:
            self.upsample = nn.Sequential(
                nn.Conv1d(n_rkhs*3 + 3, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 256, 1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, 3, 1),
                Normalize(dim=1)
            )


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    @staticmethod
    def dist2weight(dist):
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        return weight

    def forward(self, pointcloud, get_feature=False):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)
        out = []
        xyz_bank = [xyz]
        for module, prediction_modules in zip(self.SA_modules, self.prediction_modules):
            xyz, features = module(xyz, features)
            out.append(prediction_modules(features))
            xyz_bank.append(xyz)

        if not get_feature:
            if not self.point_wise_out:
                return out, torch.cat([self.adaptive_maxpool(now_out).squeeze(2) for now_out in out], dim=1)
            else:
                global_feature = torch.cat([self.adaptive_maxpool(now_out).squeeze(2) for now_out in out], dim=1)
                interpolated_feats = global_feature.unsqueeze(-1).expand(-1, -1, 1024)
                final_feature = torch.cat(
                    [xyz_bank[0].transpose(1, 2), interpolated_feats],
                    dim=1)
                point_wise_pred = self.upsample(final_feature)

                return out, global_feature, point_wise_pred
        else:
            global_feature = torch.cat([self.adaptive_maxpool(now_out).squeeze(2) for now_out in out], dim=1)
            return global_feature

