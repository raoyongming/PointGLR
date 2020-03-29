import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import os
from torchvision import transforms
from models.rscnn_ssn_ss import RSCNN_SSN, MetricLoss, ChamferLoss
from models.pointnet2_ss import PointNet2, NormalLoss

from models.foldingnet import FoldingNet, NormalNet
from data import ModelNetCls

import utils.pytorch_utils as pt_utils
import utils.pointnet2_utils as pointnet2_utils
import data.data_utils as d_utils
import argparse
import random
import yaml
from sklearn.svm import LinearSVC
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Global-Local Reasoning Training')
parser.add_argument('--config', default='cfgs/config.yaml', type=str)
parser.add_argument('--name', default='default', type=str)
parser.add_argument('--arch', default='pointnet2', type=str)
parser.add_argument('--trainset', default='modelnet', type=str)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def main():
    global svm_best_acc40
    svm_best_acc40 = 0
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    print("\n**************************")
    for k, v in config['common'].items():
        setattr(args, k, v)
        print('\n[%s]:'%(k), v)
    print("\n**************************\n")

    os.makedirs('./ckpts/', exist_ok=True)

    # dataset
    train_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])

    ss_dataset = ModelNetCls(transforms=train_transforms, self_supervision=True, use_normal=True, dataset_rate=1)
    ss_dataloader = DataLoader(
        ss_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        pin_memory=True, worker_init_fn=worker_init_fn
    )
    
    train_dataset40 = ModelNetCls(transforms=train_transforms, self_supervision=False, train=True)
    train_dataloader40 = DataLoader(
        train_dataset40,
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=int(args.workers), 
        pin_memory=True, worker_init_fn=worker_init_fn
    )

    test_dataset40 = ModelNetCls(transforms=test_transforms, self_supervision=False, train=False)
    test_dataloader40 = DataLoader(
        test_dataset40,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=int(args.workers), 
        pin_memory=True
    )

    # models
    n_rkhs = 512

    if args.arch == 'pointnet2':
        encoder = PointNet2(n_rkhs=n_rkhs, input_channels=args.input_channels, use_xyz=True, point_wise_out=True, multi=args.multiplier)
        print('Using PointNet++ backbone')
    elif args.arch == 'rscnn':
        encoder = RSCNN_SSN(n_rkhs=n_rkhs, input_channels=args.input_channels, relation_prior=args.relation_prior, use_xyz=True, point_wise_out=True, multi=args.multiplier)
        print('Using RSCNN backbone')
    else:
        raise NotImplementedError

    encoder = nn.DataParallel(encoder).cuda()
    decoer = FoldingNet(in_channel=n_rkhs * 3)
    decoer = nn.DataParallel(decoer).cuda()

    # optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoer.parameters()), lr=args.base_lr, weight_decay=args.weight_decay)

    # resume
    begin_epoch = -1
    checkpoint_name = './ckpts/' + args.name + '.pth'
    if os.path.isfile(checkpoint_name):
        checkpoint = torch.load(checkpoint_name)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoer.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        svm_best_acc40 = checkpoint['svm_best_acc40']
        begin_epoch = checkpoint['epoch'] - 1
        print("-> loaded checkpoint %s (epoch: %d)" % (checkpoint_name, begin_epoch))

    lr_lbmd = lambda e: max(args.lr_decay**(e // args.decay_step), args.lr_clip / args.base_lr)
    bnm_lmbd = lambda e: max(args.bn_momentum * args.bn_decay**(e // args.decay_step), args.bnm_clip)
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=begin_epoch)
    bnm_scheduler = pt_utils.BNMomentumScheduler(encoder, bnm_lmbd, last_epoch=begin_epoch)

    num_batch = len(ss_dataset)/args.batch_size

    args.val_freq_epoch = 1.0
    
    # training & evaluation
    train(ss_dataloader, train_dataloader40, test_dataloader40, encoder, decoer, optimizer, lr_scheduler, bnm_scheduler, args, num_batch, begin_epoch)
    

def train(ss_dataloader, train_dataloader, test_dataloader, encoder, decoer, optimizer, lr_scheduler, bnm_scheduler, args, num_batch, begin_epoch):
    PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()   # initialize augmentation
    PointcloudRotate = d_utils.PointcloudRotate()
    metric_criterion = MetricLoss()
    chamfer_criterion = ChamferLoss()
    global svm_best_acc40
    batch_count = 0
    encoder.train()
    decoer.train()

    for epoch in range(begin_epoch, args.epochs):
        np.random.seed()
        for i, data in enumerate(ss_dataloader, 0):
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)
            if bnm_scheduler is not None:
                bnm_scheduler.step(epoch-1)
            points = data
            points = Variable(points.cuda())

            # data augmentation
            sampled_points = 1200
            has_normal = (points.size(2) > 3)

            if has_normal:
                normals = points[:, :, 3:6].contiguous()
            points = points[:, :, 0:3].contiguous()

            fps_idx = pointnet2_utils.furthest_point_sample(points, sampled_points)  # (B, npoint)
            fps_idx = fps_idx[:, np.random.choice(sampled_points, args.num_points, False)]
            points_gt = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            if has_normal:
                normals = pointnet2_utils.gather_operation(normals.transpose(1, 2).contiguous(), fps_idx)
            points = PointcloudScaleAndTranslate(points_gt.data)

            # optimize
            optimizer.zero_grad()
            
            features1, fuse_global, normals_pred = encoder(points)
            global_feature1 = features1[2].squeeze(2)
            refs1 = features1[0:2]
            recon1 = decoer(fuse_global).transpose(1, 2)  # bs, np, 3

            loss_metric = metric_criterion(global_feature1, refs1)
            loss_recon = chamfer_criterion(recon1, points_gt)
            if has_normal:
                loss_normals = NormalLoss(normals_pred, normals)
            else:
                loss_normals = normals_pred.new(1).fill_(0)
            loss = loss_recon + loss_metric + loss_normals
            loss.backward()
            optimizer.step()
            if i % args.print_freq_iter == 0:
                print('[epoch %3d: %3d/%3d] \t metric/chamfer/normal loss: %0.6f/%0.6f/%0.6f \t lr: %0.5f' % (epoch+1, i, num_batch, loss_metric.item(), loss_recon.item(), loss_normals.item(), lr_scheduler.get_lr()[0]))
            batch_count += 1
            
            # validation
            if args.evaluate and batch_count % int(args.val_freq_epoch * num_batch) == 0:
                svm_acc40 = validate(train_dataloader, test_dataloader, encoder, args)

                save_dict = {'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
                             'optimizer_state_dict': optimizer.state_dict(),
                             'encoder_state_dict': encoder.state_dict(),
                             'decoder_state_dict': decoer.state_dict(),
                             'svm_best_acc40': svm_best_acc40,
                             }
                checkpoint_name = './ckpts/' + args.name + '.pth'
                torch.save(save_dict, checkpoint_name)
                if svm_acc40 == svm_best_acc40:
                    checkpoint_name = './ckpts/' + args.name + '_best.pth'
                    torch.save(save_dict, checkpoint_name)


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]


def adjust_learning_rate(optimizer, epoch, args):
    step = int(epoch // 20)
    lr = args.base_lr * (0.7 ** step)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def validate(train_dataloader, test_dataloader, encoder, args):
    global svm_best_acc40
    encoder.eval()

    test_features = []
    test_label = []

    train_features = []
    train_label = []

    PointcloudRotate = d_utils.PointcloudRotate()

    # feature extraction
    with torch.no_grad():
        for j, data in enumerate(train_dataloader, 0):
            points, target = data
            points, target = points.cuda(), target.cuda()

            num_points = 1024

            fps_idx = pointnet2_utils.furthest_point_sample(points, num_points)  # (B, npoint)
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()

            feature = encoder(points, get_feature=True)
            target = target.view(-1)

            train_features.append(feature.data)
            train_label.append(target.data)

        for j, data in enumerate(test_dataloader, 0):
            points, target = data
            points, target = points.cuda(), target.cuda()

            fps_idx = pointnet2_utils.furthest_point_sample(points, args.num_points)  # (B, npoint)
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()

            feature = encoder(points, get_feature=True)
            target = target.view(-1)
            test_label.append(target.data)
            test_features.append(feature.data)

        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)

    # train svm
    svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(), test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

    if svm_acc > svm_best_acc40:
        svm_best_acc40 = svm_acc

    encoder.train()
    print('ModelNet 40 results: svm acc=', svm_acc, 'best svm acc=', svm_best_acc40)
    print(args.name, args.arch)

    return svm_acc


if __name__ == "__main__":
    main()