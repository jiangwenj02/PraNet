import os
import os.path as osp
import numpy as np
import json
import PIL.Image
import PIL.ImageDraw
from tqdm import tqdm
 
def shape_to_mask(img_shape, points, category_id):
    """Cenerates a mask of input information corresponding to an image
    Args:
        img_shape (tuple): size of mask (width, height)
        points (list(list)): points of the shape [[x1, y1, x2, y2, .....], [x1, y1, x2, y2, .....], ...]
        category_id (list(list)): shapes category [int, int, ...]
    Returns:
        np.array(bool): mask
    """
    img_mask = np.zeros(img_shape, dtype=np.uint8)
    for i in range(len(category_id)):
        shape_mask = np.zeros(img_shape, dtype=np.uint8)
        shape_mask = PIL.Image.fromarray(shape_mask)
        draw = PIL.ImageDraw.Draw(shape_mask)
        # 由coco格式解压坐标
        xy = tuple(zip(points[i][0::2], points[i][1::2]))
    
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
        # 对每个类别生成的mask进行叠加，此处要求mask不能重合，否则会出现标签溢出的现象，即叠加后产生标签值增大的现象
        img_mask = img_mask + np.array(shape_mask, dtype=np.uint8) * category_id[i]
 
    return img_mask
 
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
    annotations = json.load(open(json_file))
    # 统计总的标签类别 这个好像后面没用着....
    categories = []
    for category in annotations["categories"]:
        if category["id"] not in categories:
            categories.append(category["id"])
    # 按照每张图片进行处理
    for i in tqdm(range(len(annotations["images"]))):
        name = annotations["images"][i]["file_name"]
        w, h = annotations["images"][i]["width"], annotations["images"][i]["height"]
        
        # 取出每个shape的 id 和 点
        points, category_id = [], []
        for shape in annotations["annotations"]:
            import pdb
            #pdb.set_trace()
            #print(i, shape["image_id"], shape["image_id"] == str(i))
            if shape["image_id"] == i:
                points.extend(shape["segmentation"])
                #print(points)
                category_id.append(shape["category_id"])
            # if int(shape["image_id"]) > i:  # 早停 减少搜索数, 不确定是否按顺序排序，如果标签按照顺序排序，早停会减少处理时间
            #     break
        
        mask = shape_to_mask((h, w), points, category_id)

        file_path = os.path.join(export_dir, name)
        dir_name = osp.abspath(osp.dirname(file_path))
        mkdir_or_exist(dir_name)
        
        PIL.Image.fromarray(mask).save(file_path)
 
def main():
    cooc_to_segmentation('/data0/zzhang/new_polyp_annotation_01_03/train.json', './train_anno')      
    cooc_to_segmentation('/data0/zzhang/new_polyp_annotation_01_03/test.json', './test_anno') 
    #cooc_to_segmentation('E:/Users/jiangwenj02/Downloads/new_polyp_annotation_01_03/test.json', './test_anno') 
 
if __name__ == "__main__":
    main()