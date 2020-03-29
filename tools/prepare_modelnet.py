import numpy as np
import torch
import os, sys

path_utils = 'utils'
path_data = 'dataset/modelnet40_normal_resampled/'

sys.path.append(path_utils)
import pointnet2_utils


def fps_sample(numpy_x, num_point):
    points = torch.from_numpy(numpy_x).float().view(1, -1, 6).cuda()
    xyz = points[:, :, :3].contiguous()
    fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_point)  # (B, npoint)
    points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
    points = points.data.cpu().numpy()
    return points[0]

root = path_data
catfile = os.path.join(root, 'modelnet40_shape_names.txt')
cat = [line.rstrip() for line in open(catfile)]
classes = dict(zip(cat, range(len(cat))))
shape_ids = {}
shape_ids['train'] = [line.rstrip() for line in open(os.path.join(root, 'modelnet40_train.txt'))]
shape_ids['test'] = [line.rstrip() for line in open(os.path.join(root, 'modelnet40_test.txt'))]

splits = ['train', 'test']

for split in splits:
    shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
    datapath = [(shape_names[i], os.path.join(root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                in range(len(shape_ids[split]))]
    print(split, len(datapath))
    out_points = np.zeros((len(datapath), 2048, 6), dtype=np.float32)
    out_labels = np.zeros((len(datapath),), dtype=int)
    for index in range(len(datapath)):
        fn = datapath[index]
        label = classes[datapath[index][0]]
        out_labels[index] = label
        points_np = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
        points = fps_sample(points_np, 2048)
        out_points[index] = points
        print(fn, points_np.shape, points.shape, index)
    np.save('./dataset/ModelNet40_normal_2048_' + split + '_points.npy', out_points)
    np.save('./dataset/ModelNet40_normal_2048_' + split + '_label.npy', out_labels)

