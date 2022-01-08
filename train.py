import argparse
import logging
import os
import sys
import numpy as np
from torch import optim
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils.dataset_2 import BasicDataset
from torch.utils.data import DataLoader, random_split
from predict_fwiou import predict_fwiou
import torch.nn.functional as F
from network import MSResNet

dir_img = ''
dir_mask = ''
dir_checkpoint = 'checkpoints/'

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    assert alpha > 0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]

    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


def rot_img(x, theta, dtype):
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid)
    return x

def rot_img_cuda(x, theta, dtype):
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    grid =grid.to(device='cuda')
    x = F.grid_sample(x, grid)
    return x

def train_net(net,
              device,
              epochs=30,
              batch_size=1,
              lr=0.0001,
              val_percent=0.1,
              save_cp=True,
              img_scale=1.0):

    dataset = BasicDataset(dir_img, dir_mask, img_scale,data_agu=False)
    n_val=200 #no use
    n_train = len(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,drop_last=False)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0
    best_miou = 0.0  # 优选超参数时用
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer=optim.Adam(net.parameters(),lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=15)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                masks_probs=torch.sigmoid(masks_pred)

                ##########################SSL:Rotation Transformation Learning####################################
                imgs_rot90 = rot_img_cuda(imgs, np.pi / 2, dtype=torch.float32)  
                imgs_rot180 = rot_img_cuda(imgs, np.pi, dtype=torch.float32)
                imgs_rot270 = rot_img_cuda(imgs, (np.pi / 2) * 3, dtype=torch.float32)
                masks_pred_rot90 = net(imgs_rot90)
                masks_pred_rot180 = net(imgs_rot180)
                masks_pred_rot270 = net(imgs_rot270)
                masks_pred_rot90 = rot_img_cuda(masks_pred_rot90, (np.pi / 2) * 3, dtype=torch.float32)  
                masks_pred_rot180 = rot_img_cuda(masks_pred_rot180, np.pi, dtype=torch.float32)
                masks_pred_rot270 = rot_img_cuda(masks_pred_rot270, np.pi / 2, dtype=torch.float32)
                masks_pred_rot90_sig = torch.sigmoid(masks_pred_rot90)
                masks_pred_rot180_sig = torch.sigmoid(masks_pred_rot180)
                masks_pred_rot270_sig = torch.sigmoid(masks_pred_rot270)
                rot90_consistency_loss = torch.mean((masks_probs - masks_pred_rot90_sig) ** 2)  
                rot180_consistency_loss = torch.mean((masks_probs - masks_pred_rot180_sig) ** 2)
                rot270_consistency_loss = torch.mean((masks_probs - masks_pred_rot270_sig) ** 2)

                ##########################SSL:Flip Transformation Learning####################################
                # horflip
                horflip_imgs = flip(imgs, 2)  
                masks_pred_horflip = net(horflip_imgs)
                masks_pred_horflip = flip(masks_pred_horflip, 2)
                masks_pred_horflip_sig = torch.sigmoid(masks_pred_horflip)
                horflip_loss = torch.mean((masks_probs - masks_pred_horflip_sig) ** 2)
                # verflip
                verflip_imgs = flip(imgs, 3)  
                masks_pred_verflip = net(verflip_imgs)
                masks_pred_verflip = flip(masks_pred_verflip, 3)
                masks_pred_verflip_sig = torch.sigmoid(masks_pred_verflip)
                verflip_loss = torch.mean((masks_probs - masks_pred_verflip_sig) ** 2)

                ###########################SSL:Noise Disturbance Learning################################
                #noise = torch.clamp(torch.randn_like(imgs) * 0.1, -0.2, 0.2)
                #imgs_noise = imgs + noise
                #masks_pred_noise = net(imgs_noise)
                #masks_pred_noise_sig = torch.sigmoid(masks_pred_noise)
                #noise_loss = torch.mean((masks_probs - masks_pred_noise_sig) ** 2)

                #########################SSL:Image Resolution Learning##################################
                #upsample = nn.Upsample(scale_factor=0.5, mode='nearest')
                #imgs_up = upsample(imgs)  
                #masks_pred_up = net(imgs_up)
                #downsample = nn.Upsample(scale_factor=2, mode='nearest')
                #masks_pred_up = downsample(masks_pred_up)
                #masks_pred_up_sig = torch.sigmoid(masks_pred_up)
                #u2d_loss = torch.mean((masks_probs - masks_pred_up_sig) ** 2)

                #######################################SSL:Image Context Fusion Learning#########################
                #lam = np.random.beta(1., 1.)
                # rand_index = torch.randperm(imgs.size()[0]).cuda()
                #rand_index = torch.randperm(imgs.size()[1])
                #bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
                #imgs_cut = imgs.clone()
                # imgs_cut[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
                #imgs_cut[:, :, bbx1:bbx2, bby1:bby2] = imgs[:, rand_index, bbx1:bbx2, bby1:bby2]
                #pred_cutmix = net(imgs_cut)
                #pred_cutmix_sig = torch.sigmoid(pred_cutmix) 
                #masks_probs_cutmix = masks_probs.clone()
                # masks_probs_cutmix[:, :, bbx1:bbx2, bby1:bby2] = masks_probs[rand_index, :, bbx1:bbx2,bby1:bby2]  
                # masks_pred_cutmix_sig = torch.sigmoid(masks_pred_cutmix)
                #cutmix_loss1 = torch.mean((masks_probs_cutmix - pred_cutmix_sig) ** 2)

               ###############################SSL#########################################
                imgs2 = batch['image2']

                imgs2 = imgs2.to(device=device, dtype=torch.float32)

                masks_pred2 = net(imgs2)
                masks_probs2 = torch.sigmoid(masks_pred2)

                ##########################SSL:Rotation Transformation Learning####################################
                imgs2_rot90 = rot_img_cuda(imgs2, np.pi / 2, dtype=torch.float32) 
                imgs2_rot180 = rot_img_cuda(imgs2, np.pi, dtype=torch.float32)
                imgs2_rot270 = rot_img_cuda(imgs2, (np.pi / 2) * 3, dtype=torch.float32)
                masks_pred2_rot90 = net(imgs2_rot90)
                masks_pred2_rot180 = net(imgs2_rot180)
                masks_pred2_rot270 = net(imgs2_rot270)
                masks_pred2_rot90 = rot_img_cuda(masks_pred2_rot90, (np.pi / 2) * 3, dtype=torch.float32)  
                masks_pred2_rot180 = rot_img_cuda(masks_pred2_rot180, np.pi, dtype=torch.float32)
                masks_pred2_rot270 = rot_img_cuda(masks_pred2_rot270, np.pi / 2, dtype=torch.float32)
                masks_pred2_rot90_sig = torch.sigmoid(masks_pred2_rot90)
                masks_pred2_rot180_sig = torch.sigmoid(masks_pred2_rot180)
                masks_pred2_rot270_sig = torch.sigmoid(masks_pred2_rot270)
                rot90_consistency_loss2 = torch.mean((masks_probs2 - masks_pred2_rot90_sig) ** 2)  
                rot180_consistency_loss2 = torch.mean((masks_probs2 - masks_pred2_rot180_sig) ** 2)
                rot270_consistency_loss2 = torch.mean((masks_probs2 - masks_pred2_rot270_sig) ** 2)

                ##########################SSL:Flip Transformation Learning###########################
                # horflip
                horflip_imgs2 = flip(imgs2, 2)  
                masks_pred2_horflip = net(horflip_imgs2)
                masks_pred2_horflip = flip(masks_pred2_horflip, 2)
                masks_pred2_horflip_sig = torch.sigmoid(masks_pred2_horflip)
                horflip_loss2 = torch.mean((masks_probs2 - masks_pred2_horflip_sig) ** 2)
                # verflip
                verflip_imgs2 = flip(imgs2, 3)  
                masks_pred2_verflip = net(verflip_imgs2)
                masks_pred2_verflip = flip(masks_pred2_verflip, 3)
                masks_pred2_verflip_sig = torch.sigmoid(masks_pred2_verflip)
                verflip_loss2 = torch.mean((masks_probs2 - masks_pred2_verflip_sig) ** 2)

                #########################SSL:Noise Disturbance Learning########################
                #noise2 = torch.clamp(torch.randn_like(imgs2) * 0.1, -0.2, 0.2)
                #imgs2_noise = imgs2 + noise2
                #masks_pred2_noise = net(imgs2_noise)
                #masks_pred2_noise_sig = torch.sigmoid(masks_pred2_noise)
                #noise_loss2 = torch.mean((masks_probs2 - masks_pred2_noise_sig) ** 2)

                #########################SSL:Image Resolution Learning##########################
                #imgs2_up = upsample(imgs2)  
                #masks_pred2_up = net(imgs2_up)
                #masks_pred2_up = downsample(masks_pred2_up)
                #masks_pred2_up_sig = torch.sigmoid(masks_pred2_up)
                #u2d_loss2 = torch.mean((masks_probs2 - masks_pred2_up_sig) ** 2)

                ##############################SSL:Image Context Fusion Learning#######################
                #imgs2_cut = imgs2.clone()
                # imgs2_cut[:, :, bbx1:bbx2, bby1:bby2] = imgs2[rand_index, :, bbx1:bbx2, bby1:bby2]
                #imgs2_cut[:, :, bbx1:bbx2, bby1:bby2] = imgs2[:, rand_index, bbx1:bbx2, bby1:bby2]
                #pred_cutmix2 = net(imgs2_cut)
                #pred_cutmix2_sig = torch.sigmoid(pred_cutmix2)  
                #masks_probs2_cutmix = masks_probs2.clone()
                # masks_probs2_cutmix[:, :, bbx1:bbx2, bby1:bby2] = masks_probs2[rand_index, :, bbx1:bbx2,bby1:bby2] 
                # masks_pred_cutmix2_sig = torch.sigmoid(masks_pred2_cutmix)
                #cutmix_loss2 = torch.mean((masks_probs2_cutmix - pred_cutmix2_sig) ** 2)

                main_loss = criterion(masks_probs, true_masks)
                rot_loss=rot90_consistency_loss+rot180_consistency_loss+rot270_consistency_loss+rot90_consistency_loss2+rot180_consistency_loss2+rot270_consistency_loss2
                flip_loss=horflip_loss+verflip_loss+horflip_loss2+verflip_loss2

                loss = main_loss+8*(rot_loss+flip_loss)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'MSResNet_via_SSL-epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
            miou = predict_fwiou(epoch)
            writer.add_scalar('MIOU/val', miou, epoch + 1)
            logging.info('this epoch val MIOU: {}'.format(miou))
            if miou >= best_miou:
                best_miou = miou
                best_epoch = epoch + 1
            scheduler.step(best_miou)
            logging.info('best val MIOU: {}'.format(best_miou))
            logging.info('best epoch: {}'.format(best_epoch))

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=150,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1.0,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net = MSResNet()
    net.n_classes=1
    net.n_channels=3
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n')
                 #f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale)
                  #val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
