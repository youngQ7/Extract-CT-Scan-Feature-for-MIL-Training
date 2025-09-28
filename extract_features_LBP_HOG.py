import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_modules.dataset_LBP import Patient_CTs_Bag
from models.builder import get_encoder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 定义compute_w_loader函数
def compute_w_loader(output_path, loader, model, verbose=1):
    if verbose > 0:
        print(f'processing a total of {len(loader)} batches')
    features = []
    lbps = []

    if args.model_name in ['resnet50', 'resnet101']:
        layer = model._modules['avgpool']
        model.eval()
        with torch.no_grad():
            for batch, lbp_batch in tqdm(loader):
                batch = batch.to(device, non_blocking=True)
                batch_features = torch.zeros(batch.size(0), 2048)  # 初始化当前批次的特征

                def copy(module, input, output):  # 定义钩子函数
                    batch_features.copy_(output.squeeze(-1).squeeze(-1))

                h = layer.register_forward_hook(copy)  # 注册前向钩子
                model(batch)  # 前向传播
                h.remove()  # 移除钩子
                features.append(batch_features.cpu())
                lbps.append(lbp_batch.cpu())

    else:
        model.eval()
        with torch.no_grad():
            for batch, lbp_batch in tqdm(loader):
                batch = batch.to(device, non_blocking=True)
                batch_features = model(batch)
                features.append(batch_features.cpu())
                lbps.append(lbp_batch.cpu())

    features = torch.cat(features, dim=0)
    lbps = torch.cat(lbps, dim=0)
    print(features.shape)
    print(features)
    print(lbps.shape)
    torch.save({
        'features': features,
        'lbps': lbps
    }, output_path)

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--dataset', type=str, default='privacy')
parser.add_argument('--data_dir', type=str, default=r'E:\CQ500')
parser.add_argument('--feat_dir', type=str, default=r'Feature_dir\CQ500')
parser.add_argument('--model_name', type=str, default='resnet50',
                    choices=['resnet50_trunc', 'plip', 'vit', 'resnet50', 'resnet101', 'swinvit', 'uni_v1', 'conch_v1']) # resnet50_trunc/uni_v1/swinvit:1024, resnet50/resnet101：2048, plip:512
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--batch_size', type=int, default=256)
args = parser.parse_args()

if __name__ == '__main__':

    # 创建好目录文件
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt', args.model_name), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt', args.model_name))

    # 加载模型和图片转换器
    model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
    model = model.to(device)

    if args.dataset == 'privacy':
        patients = []
        for data_dir in [r'E:\1_1\after', r'E:\2_1\after', r'G:\三峡\2020\术后CT', r'G:\三峡\2021\术后CT',
                     r'G:\三峡\2022\术后CT', r'G:\三峡\2023\术后CT']:
            patients.extend([p for p in Path(data_dir).iterdir() if p.is_dir()])
    else:
        patients = [p for p in Path(args.data_dir).iterdir() if p.is_dir()]
    total = len(patients)

    loader_kwargs = {'num_workers': 0, 'pin_memory': True} if device.type == "cuda" else {}

    for patient in tqdm(patients, total=total):
        patient_name = patient.name
        bag_name = patient.name+'.pt'
        print('\n{}'.format(patient_name))

        if not args.no_auto_skip and patient_name + '.pt' in dest_files:
            print('skipped {}'.format(patient_name))
            continue

        output_path = os.path.join(args.feat_dir, 'pt', args.model_name, bag_name)
        time_start = time.time()

        # 定义数据集
        dataset = Patient_CTs_Bag(patient_dir=patient, img_transforms=img_transforms)
        # 定义数据加载器
        loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
        # 计算并保存特征
        compute_w_loader(output_path, loader=loader, model=model, verbose=1)

        time_end = time.time()
        print('Time elapsed: {:.2f}s'.format(time_end - time_start))