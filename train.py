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
from skimage.metrics import structural_similarity as ski_ssim
from utils import AverageMeter
from datasets.loader import PairLoader
from models.dehazeformer import *
from CNN.CNN_model import *
from loss import *
import torchvision.utils as vutils
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dehazeformer-t', type=str, help='model name')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='indoor', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

model = CNN()
model = model.to(device)
############################################################################################################################################
model_path1 = f'C:/Users/86198/PycharmProjects/pythonProject/myself/DehazeFormer-main/CNN/indoor.pth'
model.load_state_dict(torch.load(model_path1, map_location="cuda:0"),strict=False)
model.eval()


def train(train_loader, network, criterion, optimizer, scaler):
	losses = AverageMeter()

	torch.cuda.empty_cache()

	network.train()
	i = 0
	for batch in train_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		with autocast(args.no_autocast):
			out, o1, o2, o3=  model(target_img)
			# output = network(source_img)
			output,p1,p2,p3= network(source_img)
			i = i+1
			kt_loss = nn.L1Loss()
			ts1 = kt_loss(o1,p1)
			ts2 = kt_loss(o2,p2)
			ts3 = kt_loss(o3,p3)
			# tsloss =  0.1*ts1  + 0.1*ts3 + 0.1*ts2+ 0.1*ts4
			tsloss =  0.1*ts3 + 0.1*ts2+0.1*ts1
			# tsloss = 0.1 * ts3

			vutils.save_image(output.data, 'D:/wangshilong/DehazeFormer-main/results/indoor/output/output_sample%d.png' % i,
							  normalize=True)
			vutils.save_image(source_img.data, 'D:/wangshilong/DehazeFormer-main/results/indoor/input/input_sample%d.png' %i,
							  normalize=True)
			vutils.save_image(target_img.data, 'D:/wangshilong/DehazeFormer-main/results/indoor/target/target_sample%d.png' % i,
							  normalize=True)
			# per = PerceptualLoss()
			# loss = criterion(output,target_img)
			# vutils.save_image(out.data, 'D:/wangshilong/DehazeFormer-main/results/indoor/out/out_sample%d.png' % i,
			# 				  normalize=True)
			# 将输入数据转换为单精度
			loss = 1.2*criterion(output, target_img) + tsloss

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
	i = 0

	for batch in val_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		with torch.no_grad():	# torch.no_grad() may cause warning
			# out, o1, o2, o3 = model(target_img)
			output,p1,p2,p3= network(source_img)
			output = output.clamp_(-1, 1)
			i = i+1
			vutils.save_image(output.data, 'D:/wangshilong/DehazeFormer-main/results/indoor/test/out_sample%d.png' % i,
							  normalize=True)

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
		PSNR.update(psnr.item(), source_img.size(0))
		SSIM.update(test_ssim, source_img.size(0))

	return PSNR.avg,SSIM.avg


if __name__ == '__main__':
	setting_filename = os.path.join('configs', args.exp, args.model+'.json')
	if not os.path.exists(setting_filename):
		# setting_filename = os.path.join('configs', args.exp, 'default.json')
		setting_filename = os.path.join('configs', 'default.json')
	with open(setting_filename, 'r') as f:
		setting = json.load(f)

	network = eval(args.model.replace('-', '_'))()
	network = nn.DataParallel(network).cuda()

	criterion = nn.L1Loss()

	if setting['optimizer'] == 'adam':
		optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
	elif setting['optimizer'] == 'adamw':
		optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
	else:
		raise Exception("ERROR: unsupported optimizer") 

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min=setting['lr'] * 1e-2)
	scaler = GradScaler()

	dataset_dir = os.path.join(args.data_dir, args.dataset)
	train_dataset = PairLoader(dataset_dir, 'train', 'train', 
								setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
	train_loader = DataLoader(train_dataset,batch_size=setting['batch_size'],shuffle=True,num_workers=args.num_workers,)
	val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'], setting['patch_size'])
	val_loader = DataLoader(val_dataset,batch_size=setting['batch_size'],num_workers=args.num_workers,pin_memory=True)

	save_dir = os.path.join(args.save_dir, args.exp)
	os.makedirs(save_dir, exist_ok=True)

	if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):
		print('==> Start training, current model name: ' + args.model)
		# print(network)

		writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

		best_psnr = 0
		best_ssim = 0
		for epoch in tqdm(range(setting['epochs'] + 1)):
			loss = train(train_loader, network, criterion, optimizer, scaler)
			print(loss)

			writer.add_scalar('train_loss', loss, epoch)

			scheduler.step()

			if epoch % setting['eval_freq'] == 0:
				avg_psnr,avg_ssim = valid(val_loader, network)
				print(avg_psnr)
				print(avg_ssim)
				writer.add_scalar('valid_psnr', avg_psnr, epoch)

				if avg_psnr > best_psnr:
					best_psnr = avg_psnr
					torch.save({'state_dict': network.state_dict()},os.path.join(save_dir, args.model+'.pth'))
				
				writer.add_scalar('best_psnr', best_psnr, epoch)

	else:
		print('==> Existing trained model')
		exit(1)
