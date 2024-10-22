# This code is adapted from https://github.com/MrGiovanni/ModelsGenesis/blob/master/competition/Genesis_nnUNet.py. 
# The original code is licensed under the attached LICENSE (https://github.com/yuanyuan29/Z-SSMNet/blob/master/src/z_ssmnet/ssl/LICENSE).

import warnings
from pathlib import Path
from typing import Union

warnings.filterwarnings('ignore')
import os
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from z_ssmnet.ssl_read_data_from_disk.pretrain.config_zonal import Config
from z_ssmnet.ssl_read_data_from_disk.pretrain.MNet import MNet
from z_ssmnet.ssl_read_data_from_disk.pretrain.utils import CustomDataset


def pretrain(
    model_dir: Union[Path, str] = "/workdir/SSL/pretrained_weights/",
    data_dir: Union[Path, str] = "/workdir/SSL/generated_cubes",
):
    print("torch = {}".format(torch.__version__))
    model_dir = Path(model_dir)
    data_dir = Path(data_dir)

    seed = 1
    random.seed(seed)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    logs_path = os.path.join(model_dir, "Logs")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    config = Config()
    model_path = os.path.join(model_dir, config.exp_name+".model")
    if os.path.exists(model_path):
        config.weights = model_path
    config.display()

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ################### configuration for model
    num_input_channels=6
    base_num_features = 32
    num_classes = 3

    model = MNet(num_input_channels, num_classes, kn = (32, 48, 64, 80, 96), ds = False, FMU = 'sub')

    model.to(device)
    print(model)

    criterion = nn.MSELoss()   
    optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=0.9, weight_decay=0.0, nesterov=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config.patience * 0.6), gamma=0.5) 

    training_dataset = CustomDataset(data_dir / "train", config = config)
    validation_dataset = CustomDataset(data_dir / "val", config = config)

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    writer = SummaryWriter(logs_path)
    best_loss = 100000
    intial_epoch =0
    num_epoch_no_improvement = 0
    sys.stdout.flush()

    config.patience = int(np.round(config.patience * 1500 / len(training_dataset)))

    if config.weights != None:
        checkpoint=torch.load(config.weights)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        intial_epoch=checkpoint['epoch']
        print("Loading weights from ",config.weights)
    sys.stdout.flush()
    for epoch in range(intial_epoch,config.nb_epoch):
        scheduler.step(epoch)
        model.train()
        print('current lr', optimizer.param_groups[0]['lr'])
        num_iter = int(len(training_dataset)//config.batch_size)

        for i, data in enumerate(training_dataloader):
            image, gt = data[0].numpy(), data[1].numpy()
            image = np.transpose(image, (0,1,4,3,2))

            seg = image[:,3,:,:,:]
            seg_0 = np.zeros_like(seg)
            seg_1 = np.zeros_like(seg)
            seg_2 = np.zeros_like(seg)

            seg_0[seg == 0] =1
            seg_1[seg == 1] =1
            seg_2[seg == 2] =1

            image = np.concatenate((image[:,0:3,:,:,:],np.expand_dims(seg_0, axis=1), np.expand_dims(seg_1,axis=1),np.expand_dims(seg_2, axis=1)), axis=1)
            gt = np.transpose(gt, (0,1,4,3,2))[:,0:3, :,:,:]

            image, gt = torch.from_numpy(image).float(), torch.from_numpy(gt).float()
            image=image.to(device, non_blocking=True)
            gt=gt.to(device, non_blocking=True)
            pred=model(image)
            
            pred=torch.sigmoid(pred)
            loss = criterion(pred,gt)
            
            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(round(loss.item(), 2))
            if (i + 1) % 5 ==0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}'
                    .format(epoch + 1, config.nb_epoch, i + 1, num_iter, np.average(train_losses)))
                sys.stdout.flush()

        model.eval()
        print("validating....")
        for i, data in enumerate(validation_dataloader):
            x,y = data[0].numpy(), data[1].numpy()
            x = np.transpose(x, (0,1,4,3,2))
            seg = x[:,3,:,:,:]
            seg_0 = np.zeros_like(seg)
            seg_1 = np.zeros_like(seg)
            seg_2 = np.zeros_like(seg)
            seg_0[seg == 0] =1
            seg_1[seg == 1] =1
            seg_2[seg == 2] =1

            x = np.concatenate((x[:,0:3,:,:,:], np.expand_dims(seg_0, axis=1), np.expand_dims(seg_1, axis=1), np.expand_dims(seg_2, axis=1)), axis=1)
            y = np.transpose(y, (0,1,4,3,2))[:, 0:3, :,:,:]
            image,gt = torch.from_numpy(x).float(), torch.from_numpy(y).float()
            image=image.to(device, non_blocking=True)
            gt=gt.to(device, non_blocking=True)
            pred=model(image)
            pred=torch.sigmoid(pred)

            loss = criterion(pred,gt)
            valid_losses.append(loss.item())

        #logging
        train_loss=np.average(train_losses)
        valid_loss=np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        writer.add_scalars("loss", {"train_loss": train_loss,"val_loss": valid_loss}, epoch+1)
        print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch+1,valid_loss,train_loss))
        train_losses=[]
        valid_losses=[]
        if valid_loss < best_loss:
            print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
            best_loss = valid_loss
            num_epoch_no_improvement = 0
            #save model
            torch.save({
                'epoch':epoch + 1,
                'state_dict' : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            },model_path)
            print("Saving model ",model_path)

        else:
            print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,num_epoch_no_improvement))
            num_epoch_no_improvement += 1
        
        if num_epoch_no_improvement == config.patience:
            print("Early Stopping")
            break
        sys.stdout.flush()


if __name__ == "__main__":
    pretrain()
