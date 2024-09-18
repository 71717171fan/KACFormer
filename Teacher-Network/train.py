import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from CNN_model import *
from utils import AverageMeter
from utils import PairLoader
import torchvision.utils as vutils
from dehazeformer import *
from skimage.metrics import structural_similarity as ski_ssim

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='CNN', type=str, help='model name')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='nh', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def train(train_loader, network, criterion, optimizer, scaler):
    losses = AverageMeter()

    torch.cuda.empty_cache()

    network.train()

    for batch in train_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with autocast(args.no_autocast):
            output,o1,o2,o3 = network(source_img)
            loss = criterion(output, target_img)
            vutils.save_image(output.data, 'D:/ws/CNN/results/output/output_sample%d.png' , normalize=True)
            vutils.save_image(source_img.data, 'D:/ws/CNN/results/input/input_sample%d.png' , normalize=True)
            vutils.save_image(target_img.data, 'D:/ws/CNN/results/target/target_sample%d.png', normalize=True)

        losses.update(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return losses.avg


def valid(val_loader, network):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    torch.cuda.empty_cache()

    network.eval()

    for batch in val_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with torch.no_grad():  # torch.no_grad() may cause warning
            # output,o1,o2,o3 = network(source_img).clamp_(-1, 1)
            output,o1,o2,o3 = network(source_img)
            vutils.save_image(output.data, 'D:/ws/CNN/results/test/test_sample%d.png', normalize=True)
            output = output.clamp_(-1, 1)
        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        output = output.data.cpu().numpy()[0]
        output[output > 1] = 1
        output[output < 0] = 0
        output = output.transpose((1, 2, 0))
        hr_patch = target_img.data.cpu().numpy()[0]
        hr_patch[hr_patch > 1] = 1
        hr_patch[hr_patch < 0] = 0
        hr_patch = hr_patch.transpose((1, 2, 0))
        test_ssim = ski_ssim(output, hr_patch, data_range=1, multichannel=True, channel_axis=2)
        SSIM.update(test_ssim, source_img.size(0))
        PSNR.update(psnr.item(), source_img.size(0))

    return PSNR.avg,SSIM.avg


if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    network = eval(args.model.replace('-', '_'))()
    network = nn.DataParallel(network).cuda()
    ################################################################################################################
    # network = CNN()
    # network = network.to(device)
    # model_path1 = f'D:/ws/CNN/results/nh1.pth'
    # network.load_state_dict(torch.load(model_path1, map_location="cuda:0"), strict=False)

    criterion = nn.L1Loss()

    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
    else:
        raise Exception("ERROR: unsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'],
                                                           eta_min=setting['lr'] * 1e-2)
    scaler = GradScaler()

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_dataset = PairLoader(dataset_dir, 'train', 'train',
                               setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              # pin_memory=True,
                              # drop_last=True)
                              )
    val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'],
                             setting['patch_size'])
    val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size'],
                            num_workers=args.num_workers,
                            pin_memory=True)

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(os.path.join(save_dir, args.model + '.pth')):
        print('==> Start training, current model name: ' + args.model)
        # print(network)

        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

        best_psnr = 0
        for epoch in tqdm(range(setting['epochs'] + 1)):
            loss = train(train_loader, network, criterion, optimizer, scaler)

            writer.add_scalar('train_loss', loss, epoch)

            scheduler.step()

            if epoch % setting['eval_freq'] == 0:
                avg_psnr,avg_ssim = valid(val_loader, network)
                print(avg_psnr)
                print(avg_ssim)
                writer.add_scalar('valid_psnr', avg_psnr, epoch)

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    # torch.save({'state_dict': network.state_dict()},
                    #            os.path.join(save_dir, args.model + '.pth'))
                    torch.save(network.state_dict(),os.path.join(save_dir, args.model + '.pth'))
                writer.add_scalar('best_psnr', best_psnr, epoch)

    else:
        print('==> Existing trained model')
        exit(1)
