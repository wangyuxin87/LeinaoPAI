import cv2
import numpy as np
import time
import os
import pickle

from torchvision import transforms as T
from torch.utils.data import DataLoader

import infers.vision as vision
from infers.detect import MtcnnDetector, create_mtcnn_net
from utils.mtcnn_utils import convert_to_square, IoU
from dataloaders.mtcnn import MTCNNData


def gen_onet_data(pnet_model_file, rnet_model_file, use_cuda=True, vis=False):
    data_dir = '/userhome/mtcnn_dataset/'
    model_store_dir = './data/prepare/'
    anno_file = './data_prepare/WIDER_ANNO/wider_origin_anno.txt'
    prefix_path = '/gdata/WIDER/WIDER_train/images'

    pnet, rnet, _ = create_mtcnn_net(p_model_path=pnet_model_file,
                                     r_model_path=rnet_model_file,
                                     use_cuda=use_cuda)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, min_face_size=12)

    test_transforms = T.Compose([
        T.ToTensor()
    ])
    test_data = MTCNNData(train=False, list_file=anno_file,
                          transform=test_transforms, img_dir=prefix_path)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    test_dataloader = DataLoader(test_data, batch_size=1,
                                 shuffle=False, **kwargs)

    all_boxes = list()
    batch_idx = 0

    for databatch in test_dataloader:
        if batch_idx % 100 == 0:
            print("%d images done" % batch_idx)
        im = databatch
        p_boxes, p_boxes_align = mtcnn_detector.detect_pnet(im=im)
        boxes, boxes_align = mtcnn_detector.detect_rnet(im=im, dets=p_boxes_align)
        if boxes_align is None:
            all_boxes.append(np.array([]))
            batch_idx += 1
            continue
        if vis:
            rgb_im = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2RGB)
            vision.vis_two(rgb_im, boxes, boxes_align)
        all_boxes.append(boxes_align)
        batch_idx += 1

    save_path = model_store_dir
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_file = os.path.join(save_path, "detections_rnet.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    post_file, neg_file, part_file, img_list, img_idx = gen_onet_sample_data(data_dir,
                                                                             anno_file,
                                                                             save_file,
                                                                             prefix_path)
    return post_file, neg_file, part_file, img_list, img_idx


def gen_onet_sample_data(data_dir, anno_file, det_boxs_file, prefix):
    anno_store_dir = './data/prepare/'
    neg_save_dir = os.path.join(data_dir, "48/negative")
    pos_save_dir = os.path.join(data_dir, "48/positive")
    part_save_dir = os.path.join(data_dir, "48/part")

    for dir_path in [neg_save_dir, pos_save_dir, part_save_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    image_size = 48
    im_idx_list = list()
    gt_boxes_list = list()
    num_of_images = len(annotations)
    print("processing %d images in total" % num_of_images)

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_idx = os.path.join(prefix, annotation[0])
        boxes = map(float, annotation[1:])
        boxes = np.array(list(boxes), dtype=np.float32).reshape(-1, 4)
        im_idx_list.append(im_idx)
        gt_boxes_list.append(boxes)

    save_path = anno_store_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    post_save_file = os.path.join(anno_store_dir, "pos_48.txt")
    neg_save_file = os.path.join(anno_store_dir, "neg_48.txt")
    part_save_file = os.path.join(anno_store_dir, "part_48.txt")

    f1 = open(post_save_file, 'w')
    f2 = open(neg_save_file, 'w')
    f3 = open(part_save_file, 'w')
    det_handle = open(det_boxs_file, 'rb')
    det_boxes = pickle.load(det_handle)
    print(len(det_boxes), num_of_images)
    assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    # index of neg, pos and part face, used as their image names
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    img_idx = 0
    img_list = []
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        if image_done % 100 == 0:
            print("%d images done" % image_done)
        image_done += 1
        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        for box in dets:
            x_left, y_top, x_right, y_bottom = box[0:4].astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or \
                    x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            if np.max(iou) < 0.3:
                # # Iou with all gts must below 0.3
                # save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                # f2.write(save_file + ' 0\n')
                # cv2.imwrite(save_file, resized_im)

                f2.write(str(img_idx) + ' 0\n')
                img_list.append(resized_im)
                img_idx += 1

                n_idx += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(iou) >= 0.65:
                    # save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    # f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1,
                    #                                                    offset_y1,
                    #                                                    offset_x2,
                    #                                                    offset_y2))
                    # cv2.imwrite(save_file, resized_im)

                    f1.write(str(img_idx) + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1,
                                                                          offset_y1,
                                                                          offset_x2,
                                                                          offset_y2))
                    img_list.append(resized_im)
                    img_idx += 1

                    p_idx += 1

                elif np.max(iou) >= 0.4:
                    # save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    # f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1,
                    #                                                     offset_y1,
                    #                                                     offset_x2,
                    #                                                     offset_y2))
                    # cv2.imwrite(save_file, resized_im)

                    f3.write(str(img_idx) + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1,
                                                                           offset_y1,
                                                                           offset_x2,
                                                                           offset_y2))
                    img_list.append(resized_im)
                    img_idx += 1

                    d_idx += 1
    f1.close()
    f2.close()
    f3.close()

    return post_save_file, neg_save_file, part_save_file, img_list, img_idx

