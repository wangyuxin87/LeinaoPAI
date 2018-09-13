# coding: utf-8
import os
import cv2
import numpy as np
import random
import sys
import numpy.random as npr
import argparse
import config as config
import src.utils.util as utils


def gen_data(anno_file, data_dir, prefix):


    size = 48
    image_id = 0

    landmark_imgs_save_dir = os.path.join(data_dir, "48/landmark")
    if not os.path.exists(landmark_imgs_save_dir):
        os.makedirs(landmark_imgs_save_dir)

    anno_dir = config.ANNO_STORE_DIR
    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)

    landmark_anno_filename = config.ONET_LANDMARK_ANNO_FILENAME
    save_landmark_anno = os.path.join(anno_dir, landmark_anno_filename)

    f = open(save_landmark_anno, 'w')
    # dstdir = "train_landmark_few"

    with open(anno_file, 'r') as f2:
        annotations = f2.readlines()

    num = len(annotations)
    print("%d total images" % num)

    l_idx =0
    idx = 0
    # image_path bbox landmark(5*2)
    for annotation in annotations:
        # print imgPath

        annotation = annotation.strip().split(' ')

        if len(annotation) != 141:
            print(annotation[0])
            print(len(annotation))
        assert len(annotation) == 141, "each line should have 141 element"

        im_path = os.path.join(prefix, annotation[0].replace("\\", "/"))

        gt_box = map(float, annotation[1:5])
        # gt_box = [gt_box[0], gt_box[2], gt_box[1], gt_box[3]]

        gt_box = np.array(list(gt_box), dtype=np.int32)

        landmark = map(float, annotation[5:])
        landmark = np.array(list(landmark), dtype=np.float)

        img = cv2.imread(im_path)
        assert (img is not None)

        height, width, channel = img.shape
        # crop_face = img[gt_box[1]:gt_box[3]+1, gt_box[0]:gt_box[2]+1]
        # crop_face = cv2.resize(crop_face,(size,size))

        idx = idx + 1
        if idx % 100 == 0:
            print("%d images done, landmark images: %d"%(idx, l_idx))

        x1, y1, x2, y2 = gt_box

        # gt's width
        w = x2 - x1 + 1
        # gt's height
        h = y2 - y1 + 1
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue
        # random shift
        for i in range(10):
            bbox_size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)
            nx1 = int(max(x1 + w / 2 - bbox_size / 2 + delta_x, 0))
            ny1 = int(max(y1 + h / 2 - bbox_size / 2 + delta_y, 0))

            nx2 = nx1 + bbox_size
            ny2 = ny1 + bbox_size
            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])
            cropped_im = img[ny1:ny2 + 1, nx1:nx2 + 1, :]
            resized_im = cv2.resize(cropped_im, (size, size),interpolation=cv2.INTER_LINEAR)

            offset_x1 = round((x1 - nx1) / float(bbox_size), 2)
            offset_y1 = round((y1 - ny1) / float(bbox_size), 2)
            offset_x2 = round((x2 - nx2) / float(bbox_size), 2)
            offset_y2 = round((y2 - ny2) / float(bbox_size), 2)

            offset_landmark = []
            for landmark_idx in range(len(landmark)):
                if landmark_idx % 2 == 0:
                    offset_landmark.append(round((landmark[landmark_idx] - nx1) / float(bbox_size), 2))
                else:
                    offset_landmark.append(round((landmark[landmark_idx] - ny1) / float(bbox_size), 2))


            # cal iou
            iou = utils.IoU(crop_box.astype(np.float), np.expand_dims(gt_box.astype(np.float), 0))
            if iou > 0.65:
                save_file = os.path.join(landmark_imgs_save_dir, "%s.jpg" % l_idx)
                cv2.imwrite(save_file, resized_im)

                # f.write(save_file + ' -2 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n' % \
                # (offset_x1, offset_y1, offset_x2, offset_y2, \
                # offset_left_eye_x,offset_left_eye_y,offset_right_eye_x,offset_right_eye_y,offset_nose_x,offset_nose_y,offset_left_mouth_x,offset_left_mouth_y,offset_right_mouth_x,offset_right_mouth_y))

                line = []
                line.append(str(save_file))
                line.append('-2')

                line.append(str(offset_x1))
                line.append(str(offset_y1))
                line.append(str(offset_x2))
                line.append(str(offset_y2))

                for tmp in offset_landmark:
                    line.append(str(tmp))
                line.append('\n')

                line_str = ' '.join(line)
                f.write(line_str)

                l_idx += 1
    print("%d images done, landmark images: %d" % (idx, l_idx))
    f.close()




def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--traindata_store', dest='traindata_store', help='dface train data temporary folder,include 12,24,48/postive,negative,part,landmark',
                        default='/dataset/', type=str)
    parser.add_argument('--anno_file', dest='annotation_file', help='celeba dataset original annotation file',
                        default=os.path.join('./300-W_ANNO', "300w_origin_anno.txt"), type=str)
    parser.add_argument('--prefix_path', dest='prefix_path', help='annotation file image prefix root path',
                        default='', type=str)


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    gen_data(args.annotation_file, args.traindata_store, args.prefix_path)


