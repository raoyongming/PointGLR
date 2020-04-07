import torch
import torch.utils.data as data
import numpy as np
import os, sys, h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


class ScanObjectNNCls(data.Dataset):

    def __init__(
            self, transforms=None, train=True, self_supervision=False
    ):
        super().__init__()

        self.transforms = transforms

        self.self_supervision = self_supervision

        self.train = train

        root = './dataset/ScanObjectNN/main_split_nobg/'
        if self.self_supervision:
            h5 = h5py.File(root + 'training_objectdataset.h5', 'r')
            points_train = np.array(h5['data']).astype(np.float32)
            h5.close()
            self.points = points_train
            self.labels = None
        elif train:
            h5 = h5py.File(root + 'training_objectdataset.h5', 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            h5 = h5py.File(root + 'test_objectdataset.h5', 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()

        print('Successfully load ScanObjectNN with', len(self.labels), 'instances')

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

