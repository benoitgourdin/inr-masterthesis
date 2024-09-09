import os
import sys
import time
import shutil
from pathlib import Path
import torch
import wandb
import numpy as np

from utils import dataloader, config, loss, file_creator
from model import implicits


def train_one_epoch(ds_loader, net, optimizer, criterion, metric, device, epoch, num_epochs_target,
                    global_step, log_epoch_count, logger, train_function):
    loss_running = 0.0
    num_losses = 0
    metric_running = 0.0
    num_metrics = 0
    coords = torch.Tensor()
    registration_pred = torch.Tensor()
    registration_gt = torch.Tensor()
    t0 = time.time()
    net.train()
    pc_writer = file_creator.PointCloudWriter()
    for batch in ds_loader:
        registration_gt = batch['registration_flow'].to(device)
        optimizer.zero_grad()
        coords = batch['moving_pc'].to(device)
        registration_pred = net(coords)
        loss = criterion(registration_pred, registration_gt, coords.squeeze())
        loss.backward()
        optimizer.step()
        loss_running += loss.item()
        num_losses += 1
        metric_running += metric(registration_pred, registration_gt, coords).item()
        num_metrics += batch['registration_flow'].shape[0]
        global_step += 1
    if epoch % log_epoch_count == 0:
        loss_avg = loss_running / num_losses
        metric_avg = metric_running / num_metrics
        num_epochs_trained = epoch + 1
        if logger is not None:
            logger.log({f"loss ({logger.config.loss_function})": loss_avg, "global_step": num_epochs_trained})
            logger.log({f"metric/train ({train_function})": metric_avg, "global_step": num_epochs_trained})
        epoch_duration = time.time() - t0
        print(f'[{num_epochs_trained}/{num_epochs_target}] '
              f'Avg loss: {loss_avg:.4f}; '
              f'metric: {metric_avg:.3f}; '
              f'global step nb. {global_step} '
              f'({epoch_duration:.1f}s)')
    if epoch == 0:
        moving_point_cloud = coords
        target_point_cloud = coords + registration_gt
        moving_point_cloud_np = moving_point_cloud.squeeze().cpu().detach().numpy()
        new_column = np.full((moving_point_cloud_np.shape[0], 1), 1)
        moving_point_cloud_np = np.hstack((moving_point_cloud_np, new_column))
        target_point_cloud_np = target_point_cloud.squeeze().cpu().detach().numpy()
        new_column = np.full((target_point_cloud_np.shape[0], 1), 14)
        target_point_cloud_np = np.hstack((target_point_cloud_np, new_column))
        logger.log({"target point cloud": wandb.Object3D(target_point_cloud_np)})
        logger.log({"moving point cloud": wandb.Object3D(moving_point_cloud_np)})
        # pc_writer.write_ply_file(target_point_cloud, '/home/mil/gourdin/inr_3d_data/target_pc.ply')
        # pc_writer.write_ply_file(moving_point_cloud, '/home/mil/gourdin/inr_3d_data/moving_pc.ply')

    if epoch % (log_epoch_count * 10) == 0:
        moved_point_cloud = coords + registration_pred
        moved_point_cloud_np = moved_point_cloud.squeeze().cpu().detach().numpy()
        new_column = np.full((moved_point_cloud_np.shape[0], 1), 5)
        moved_point_cloud_np = np.hstack((moved_point_cloud_np, new_column))
        logger.log({"moved point cloud": wandb.Object3D(moved_point_cloud_np)})
        # pc_writer.write_ply_file(moved_point_cloud, '/home/mil/gourdin/inr_3d_data/moved_pc.ply')


def validate(ds_loader, net, metric, device, epoch, logger, val_function):
    metric_running = 0.0
    num_metrics = 0
    t0 = time.time()
    net.eval()
    with torch.no_grad():
        for batch in ds_loader:
            registration_gt = batch['registration_flow'].to(device)
            coords = batch['moving_pc'].to(device)
            registration_pred = net(coords)
            metric_running += metric(registration_pred, registration_gt, coords).item()
            num_metrics += batch['registration_flow'].shape[0]
        metric_avg = metric_running / num_metrics
    if logger is not None:
        logger.log({f"metric/val ({val_function})": metric_avg, "global_step": (epoch + 1)})
    t1 = time.time()
    val_duration = t1 - t0
    print(f'[val] metric {metric_avg:.3f} ({val_duration:.1f}s)')


def main():
    params = config.parse_config()
    run = wandb.init()
    model_dir = Path(os.path.join(str(os.path.dirname(os.path.abspath(__file__))),
                                  'results', params['model_name'], run.name))\
        if params['model_name'] is not None else None
    # config file parameters
    num_epochs_target = params['num_epochs']
    log_epoch_count = params['log_epoch_count']
    checkpoint_epoch_count = params['checkpoint_epoch_count']
    max_num_checkpoints = params['max_num_checkpoints']
    train_function = params["train_metric"]
    val_function = params["val_metric"]
    # wandb parameters
    learning_rate = wandb.config.learning_rate
    num_points_per_example = wandb.config.sample_batch_share
    loss_function = wandb.config.loss_function
    optimizer_param = wandb.config.optimizer
    if model_dir is not None:
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        sys.stdout = file_creator.Logger(model_dir / 'log.txt', 'a')
        checkpoint_writer = file_creator.RollingCheckpointWriter(model_dir, 'checkpoint',
                                                                 max_num_checkpoints, 'pth')
    else:
        print('Warning: no model name provided; not writing anything to the file system.')
        checkpoint_writer = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print('Warning: no GPU available; training on CPU.')
    ds_loader_train = dataloader.create_data_loader(params['data_path'], num_points_per_example)
    ds_loader_val = dataloader.create_data_loader(params['data_path'], num_points_per_example)
    if not ds_loader_train:
        raise ValueError(f'Number of training examples is smaller than the batch size.')
    net = implicits.create_model(params, wandb)
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs for training.')
        net = torch.nn.DataParallel(net)
    net = net.to(device)
    print(net)
    if optimizer_param == "adam":
        optimizer = torch.optim.Adam([
            {'params': net.parameters(), 'lr': learning_rate}
        ])
    elif optimizer_param == "sgd":
        optimizer = torch.optim.SGD([
            {'params': net.parameters(), 'lr': learning_rate}
        ])
    else:
        optimizer = torch.optim.Adam([
            {'params': net.parameters(), 'lr': learning_rate}
        ])
    criterion = loss.create_loss(loss_function).to(device)
    train_metric = loss.create_loss(train_function).to(device)
    val_metric = loss.create_loss(val_function).to(device)
    global_step = torch.tensor(0, dtype=torch.int64)
    num_epochs_trained = 0
    for epoch in range(num_epochs_trained, num_epochs_target):
        train_one_epoch(ds_loader_train, net, optimizer, criterion, train_metric, device, epoch, num_epochs_target,
                        global_step, log_epoch_count, wandb, train_function)
        if epoch % log_epoch_count == 0:
            validate(ds_loader_val, net, val_metric, device, epoch, wandb, val_function)
        if checkpoint_writer is not None and epoch % checkpoint_epoch_count == 0:
            checkpoint_writer.write_rolling_checkpoint(
                {'net': net.state_dict()},
                optimizer.state_dict(), int(global_step.item()), epoch + 1)
    if checkpoint_writer is not None:
        checkpoint_writer.write_rolling_checkpoint(
            {'net': net.state_dict()},
            optimizer.state_dict(), int(global_step.item()), num_epochs_target)
    wandb.finish()


if __name__ == '__main__':
    wandb.login()
    sweep_configuration = {
        "method": "grid",
        "metric": {"goal": "minimize", "name": "loss"},
        "parameters": {
            "learning_rate": {"values": [0.5e-1, 0.5e-2, 0.5e-3, 0.5e-4]},
            #"latent_learning_rate": {"values": [1.0e-3]},
            "sample_batch_share": {"values": [0.1, 0.5, 1.0]},
            "loss_function": {"values": ["pointpwc"]},
            "optimizer": {"values": ["adam", "sgd"]},
            "dropout": {"values": [0.0, 0.5]},
            "hidden_layer_size": {"values": [128]},
            "activation_function": {"values": ["sine", "relu"]},
            "encoding": {"values": [False, True]},
        },
    }
    params = config.parse_config()
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=params['model_name'])
    wandb.agent(sweep_id, function=main, count=100)
