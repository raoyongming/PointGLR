import torch
import torch.utils.data as data
import numpy as np
import os, sys, h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)



class ScanNetCls(data.Dataset):

    def __init__(
            self, transforms=None, train=True, self_supervision=False
    ):
        super().__init__()

        self.transforms = transforms

        self.self_supervision = self_supervision

        self.train = train

        root = './dataset/scannet/'

        def load_h5(name_list):
            out_data = []
            out_label = []
            for name in name_list:
                h5 = h5py.File(root + name, 'r')
                points_train = np.array(h5['data']).astype(np.float32)[:, :, :3]
                labels_train = np.array(h5['label']).astype(int)
                out_data.append(points_train)
                out_label.append(labels_train)
                h5.close()
            points = np.concatenate(out_data, 0)
            labels = np.concatenate(out_label, 0)
            return points, labels

        train_list = [name.strip() for name in open(root + 'train_files.txt', 'r').readlines()]
        test_list = [name.strip() for name in open(root + 'test_files.txt', 'r').readlines()]

        if self.self_supervision:
            points_train, labels_train = load_h5(train_list)
            self.points = points_train
            self.labels = None
        elif train:
            points_train, labels_train = load_h5(train_list)
            self.points = points_train
            self.labels = labels_train
        else:
            points_train, labels_train = load_h5(test_list)
            self.points = points_train
            self.labels = labels_train

        print('Successfully load ScanNet with', self.points.shape, 'instances')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        if self.train:
            np.random.shuffle(pt_idxs)
        
        current_points = self.points[idx, pt_idxs].copy()
        
        if self.transforms is not None:
            current_points = self.transforms(current_points)

        if self.self_supervision:
            return current_points
        else:
            label = self.labels[idx]
            return current_points, label

    def __len__(self):
        return self.points.shape[0]


