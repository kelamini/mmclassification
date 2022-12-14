import os
import os.path as osp
from pathlib import Path
import json
import glob
import re
import cv2 as cv
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser

from mmcls.apis import inference_model, init_model


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def make_save_path(savedir, name="exp"):
    save_dir = increment_path(Path(savedir) / name, exist_ok=False)
    print(f"The resault save to {save_dir}\n")

    return save_dir


def save_new(savedir, imgpath, shapes, imgshape):
    save_dir = osp.join(savedir, "new")
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    
    savepath = osp.join(save_dir, osp.basename(imgpath).replace("jpg", "json"))

    new_set = {
        "version": "",
        "flags": {},
        "shapes": shapes,
        "imagePath": imgpath,
        "imageData": None,
        "imageWidth": imgshape[1],
        "imageHeight": imgshape[0]
    }
    # print(f"save new: {savepath}")
    with open(savepath, "w", encoding="utf8") as fp:
        json.dump(new_set, fp, indent=2)


def load_label(label_path):
    with open(label_path, "r", encoding="utf8") as fp:
        label_data = json.load(fp)

    return label_data


def main(args):
    img_dir = args.img_dir
    label_dir = args.label_dir
    save_dir = args.save_dir
    config = args.config
    checkpoint = args.checkpoint
    score_thr = args.score_thr
    device = args.device

    savedir = make_save_path(save_dir)
    os.makedirs(savedir)
    
    # build the model from a config file and a checkpoint file
    model = init_model(config, checkpoint, device=device)
    person_list = []

    label_list = sorted(os.listdir(label_dir))
    for filename in label_list:
        img_path = osp.join(img_dir, filename.replace("json", "jpg"))
        img = cv.imread(img_path)
        imgshape = img.shape
        det = deepcopy(img)

        label_path = osp.join(label_dir, filename)
        label_data = load_label(label_path)
        shapes = label_data["shapes"]
        new_shapes = []
        for shape in shapes:
            point = shape["points"]
            point = [int(i) for i in np.array(point).flatten()]
            img_ROI = np.array(img[point[1]:point[3], point[0]:point[2]], dtype=np.uint8)

            # test a single image
            result = inference_model(model, img_ROI)
            print(result)
            if result["pred_score"]>score_thr and result["pred_class"] in person_list:
                new_shapes.append(shapes)

                # break
        if len(new_shapes) != 0:
            save_new(savedir, img_path, new_shapes, imgshape)

            # cv.rectangle(det, (point[0], point[1]), (point[2], point[3]), (0, 0, 255), 2)
            # cv.imshow("img", img_ROI)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            
        break


def get_args():
    parser = ArgumentParser()
    
    parser.add_argument('--img_dir', type=str, default="/home/kelaboss/eval_data/xcyuan/tmp/172.25.188.3/2022_11_21_17_07_05", help='')
    parser.add_argument('--label_dir', type=str, default="/home/kelaboss/eval_data/xcyuan/tmp/172.25.188.3/deoverlap_yolov5_yolox/exp/new", help='')
    parser.add_argument('--save_dir', type=str, default="/home/kelaboss/eval_data/xcyuan/tmp/172.25.188.3/classifi_yolov5_yolox", help='')
    parser.add_argument('--config', type=str, default="configs/resnest/resnest269_64xb32_in1k.py", help='Config file')
    parser.add_argument('--checkpoint', type=str, default="weights/resnest269_imagenet_converted-59930960.pth", help='Checkpoint file')
    parser.add_argument('--score_thr', type=float, default=0.3, help='')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference, such as cuda:0')
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    opts = get_args()
    main(opts)
