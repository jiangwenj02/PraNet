import os
import os.path as osp
import numpy as np
import json
import PIL.Image
import PIL.ImageDraw
from tqdm import tqdm
from pycocotools.coco import COCO
 
def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)
    
def cooc_to_segmentation(json_file, export_dir):
    """Generate the annotation information according to the json_file
    Args:
        json_file (string): jsonfile of coco format annotations
        export_dir (string): output path
    """

    coco = COCO(json_file)
    img_ids = coco.getImgIds()
    img_infos = []
    for i in tqdm(img_ids):
        info = coco.loadImgs([i])[0]
        anns_ids = coco.getAnnIds(imgIds=i,  iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        info['filename'] = info['file_name']
        img_infos.append(info)
        anns_img = np.zeros((info['height'],info['width'])).astype(np.uint8)
        for ann in anns:
            anns_img = np.maximum(anns_img, coco.annToMask(ann)*ann['category_id'])

        file_path = os.path.join(export_dir, info['filename'])
        dir_name = osp.abspath(osp.dirname(file_path))
        mkdir_or_exist(dir_name)        
        PIL.Image.fromarray(anns_img).save(file_path)
 
def main():
    #cooc_to_segmentation('/data0/zzhang/new_polyp_annotation_01_03/train.json', './train_anno')
    #cooc_to_segmentation('/data0/zzhang/new_polyp_annotation_01_03/test.json', './test_anno') 
    cooc_to_segmentation('E:/Users/jiangwenj02/Downloads/new_polyp_annotation_01_03/test.json', './test_anno') 
 
if __name__ == "__main__":
    main()