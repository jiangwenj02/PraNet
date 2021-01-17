import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.PraNet_Res2Net import PraNet
from utils.dataloader import test_dataset
import os.path as osp
import tqdm

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet_Res2Net/PraNet-19.pth')
parser.add_argument('--json_file', type=str,
                        default='/data0/zzhang/new_polyp_annotation_01_03/test.json')

for _data_name in ['CVC-300']:
    data_path = '/data2/dataset/cleaned_data/'
    save_path = '/data0/zzhang/tmp/pranet/'
    opt = parser.parse_args()
    model = PraNet()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = data_path
    gt_root = './test_anno/'
    test_loader = test_dataset(image_root, gt_root, opt.json_file, opt.testsize)

    for i in tqdm(range(test_loader.size)):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res5, res4, res3, res2 = model(image)
        res = res2

        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze() >= 0.5
        res = res.astype(np.uint8)
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        file_path = save_path+name
        dir_name = osp.abspath(osp.dirname(file_path))
        mkdir_or_exist(dir_name)
        misc.imsave(file_path, res)