import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import ssim, psnr
from torch.utils.data import DataLoader
from collections import OrderedDict
import time  # 添加时间模块

from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dehazeformer-t', type=str, help='model name')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='indoor', type=str, help='dataset name')
parser.add_argument('--dataset1', default='nh', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
args = parser.parse_args()


def single(save_dir):
    state_dict = torch.load(save_dir)['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    return new_state_dict


def test(test_loader, network, result_dir):
    PSNR = AverageMeter()
    SSIM = AverageMeter()

    torch.cuda.empty_cache()
    network.eval()

    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
    f_result = open(os.path.join(result_dir, 'results.csv'), 'w')
    psnrs = []
    ssims = []

    start_time = time.time()  # 记录开始时间

    for idx, batch in enumerate(test_loader):
        input = batch['source'].cuda()
        target = batch['target'].cuda()

        filename = batch['filename'][0]

        with torch.no_grad():
            output, p1, p2, p3 = network(input)
            output = output.clamp_(-1, 1)

            # [-1, 1] to [0, 1]
            output = output * 0.5 + 0.5
            target = target * 0.5 + 0.5

            ssims.append(ssim(output, target).item())
            psnrs.append(psnr(output, target))

        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(result_dir, 'imgs', filename), out_img)

    avg_ssim = np.mean(ssims)
    avg_psnr = np.mean(psnrs)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算总时间

    print("平均ssim：{}".format(avg_ssim))
    print("平均psnr：{}".format(avg_psnr))
    print("测试时间：{:.2f} 秒".format(elapsed_time))  # 打印测试时间

    # f_result.close()
    #
    # os.rename(os.path.join(result_dir, 'results.csv'),
    #           os.path.join(result_dir, '%.02f | %.04f.csv'%(PSNR.avg, SSIM.avg)))


if __name__ == '__main__':
    network = eval(args.model.replace('-', '_'))()
    total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(total_params))
    network.cuda()
    saved_model_dir = os.path.join(args.save_dir, args.exp, args.model + '.pth')

    if os.path.exists(saved_model_dir):
        print('==> Start testing, current model name: ' + args.model)
        network.load_state_dict(single(saved_model_dir))
    else:
        print('==> No existing trained model!')
        exit(0)

    dataset_dir = os.path.join(args.data_dir, args.dataset1)
    test_dataset = PairLoader(dataset_dir, 'test', 'test')
    test_loader = DataLoader(test_dataset,
                            batch_size=1,
                            num_workers=args.num_workers,
                            pin_memory=True)

    result_dir = os.path.join(args.result_dir, args.dataset, args.model)
    test(test_loader, network, result_dir)
