import numpy as np
from torch.utils.data import Dataset


class HumanDataset(Dataset):
    def __init__(self, datas, labels, transform=None, target_transform=None):
        self.labels = labels
        self.datas = datas
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.datas[idx]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label


def split_data(subjects, dataset, keypoints):
    x_list, y_list, cam_list = [], [], []
    s_a = []
    # for subject in dataset.subjects():
    for subject in subjects:
        for action in dataset[subject].keys():
            y_list.append(dataset[subject][action]['positions'])
            x_list.append(np.concatenate(keypoints[subject][action], axis=2))
            tem = []
            s_a.append((subject, action))
            for cam in dataset[subject][action]['cameras']:
                # tem = []
                tem += cam['orientation'].tolist()
                tem += cam['translation'].tolist()
                tem += [cam['res_w'], cam['res_h']]
                tem += cam['intrinsic'].tolist()
                # tem_list.append(tem)
            # tem *= len(dataset[subject][action]['positions'])
            tem_list = [tem for _ in range(len(dataset[subject][action]['positions']))]
            cam_list.append(np.array(tem_list))
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    cam = np.concatenate(cam_list, axis=0)
    cam = np.expand_dims(cam, axis=1).repeat(17, axis=1)
    y = y * 1000
    y = y - y[:, : 1, :]
    # x = x.reshape(x.shape[0], -1)
    # y = y.reshape(y.shape[0], -1)
    # cam = cam.reshape(cam.shape[0], -1)
    return x, y, cam

def normalation(train, test):
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    train = (train - mean) / std
    test = (test - mean) / std
    return train, test
 
def add_noise(x, mu=0.0, sigma=1.0):
    noise = np.random.normal(mu, sigma, size=x.shape)
    return x + noise