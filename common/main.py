import numpy as np
import argparse
from common.h36m_dataset import Human36mDataset
from common.data_utils import split_data
from common.data_utils import normalation
from common.data_utils import HumanDataset
from common.model import Net
from tqdm import tqdm
from common.loss import mpjpe
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
from common.Logger import MyLogger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_3d', type=str, default='data/data_3d_h36m.npz')
    parser.add_argument('--data_2d', type=str, default='data/data_2d_h36m_gt.npz')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)

    return parser.parse_args()

def main():

    args = parse_args()
    dataset_path = args.data_3d
    dataset = Human36mDataset(dataset_path)
    keypoints_path = args.data_2d
    keypoints = np.load(keypoints_path, allow_pickle=True)
    # keypoints_metadata = keypoints['metadata'].item()
    keypoints = keypoints['positions_2d'].item()

    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in dataset[subject].keys():
            assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
            if 'positions' not in dataset[subject][action]:
                continue
            for cam_idx in range(len(keypoints[subject][action])):
                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]['positions'].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]
        
    subjects = keypoints.keys()
    for subject in subjects:
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                # kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps
    train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
    test_subjects = [ 'S9', 'S11']
    x_train, y_train, cam_train = split_data(train_subjects, dataset, keypoints)
    x_test, y_test, cam_test = split_data(test_subjects, dataset, keypoints)
    x_train, x_test = normalation(x_train, x_test)
    # print(x_train.dtype)
    # x_train = x_train[:, :, : 2]
    # x_test = x_test[:, :, : 2]
    learning_rate =args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    logger = MyLogger()
    model = Net(keypoints_num=17, n=3).cuda()
    logger.info(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    train_data = HumanDataset(x_train, y_train)
    test_data = HumanDataset(x_test, y_test)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    writer = SummaryWriter(log_dir='logs', comment='mpjpe')
    num_train = len(train_dataloader.dataset)
    num_test = len(test_dataloader.dataset)
    logger.info(f'Training data:  {num_train}    Test data:  {num_test}')
    best_loss = 1000
    for i in range(epochs):
        loss_train = 0
        loss_test = 0
        N = 0
        model.train()
        train_loop = tqdm(train_dataloader)
        for x, y in train_loop:
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            loss = mpjpe(pred, y)
            loss_train += loss.item() * y.shape[0] 
            N += y.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loop.set_description(f'Epoch {i + 1}/{epochs}')
            train_loop.set_postfix(step_loss=loss.item(), total_loss=loss_train / N)
        loss_train /= N
        N = 0
        with torch.no_grad():
            model.eval()
            test_loop = tqdm(test_dataloader)
            for x, y in test_loop:
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                pred = model(x)
                loss = mpjpe(pred, y)
                loss_test += loss.item() * y.shape[0]
                N += y.shape[0] 
                test_loop.set_description(f'Epoch {i + 1}/{epochs}')
                test_loop.set_postfix(step_loss=loss.item(), total_loss=loss_test / N)
        loss_test /= N
        logger.info(f'Epoch.  {i + 1}     Train_Loss  {loss_train}    Test_loss.  {loss_test}')
        if loss_test < best_loss:
            logger.info('save best model and the best loss is {}'.format(loss_test))
            best_loss = loss_test
            torch.save(model.state_dict(), 'best_model.pth')
        loss = {'train_loss': loss_train, 'test_loss': loss_test}
        writer.add_scalars('loss', loss, i + 1)
    logger.info('The best loss is {}'.format(best_loss))
    model.eval()
    writer.add_graph(model, x)
    writer.close()

if __name__ == '__main__':
    main()