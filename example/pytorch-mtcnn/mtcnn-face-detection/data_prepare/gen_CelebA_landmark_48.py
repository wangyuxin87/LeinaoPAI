import os
import cv2
import numpy as np
import numpy.random as npr
import utils.mtcnn_utils as utils


def gen_data(img_list, img_idx):
    data_dir = '/userhome/mtcnn_dataset/'
    anno_store_dir = './data/prepare/'
    anno_file = './data_prepare/CelebA_ANNO/landmark_anno.txt'
    prefix_path = '/gdata'

    size = 48
    # landmark_imgs_save_dir = os.path.join(data_dir, "48/landmark")
    # if not os.path.exists(landmark_imgs_save_dir):
    #     os.makedirs(landmark_imgs_save_dir)
    if not os.path.exists(anno_store_dir):
        os.makedirs(anno_store_dir)
    save_landmark_anno = os.path.join(anno_store_dir, "landmark_48.txt")
    f = open(save_landmark_anno, 'w')
    with open(anno_file, 'r') as f2:
        annotations = f2.readlines()
    num = len(annotations)
    print("%d total images" % num)
    l_idx = 0
    idx = 0
    # image_path bbox landmark(5*2)
    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        if len(annotation) != 15:
            print(annotation[0])
            print(len(annotation))
        assert len(annotation) == 15, "each line should have 15 element"
        im_path = os.path.join(prefix_path, annotation[0].replace("\\", "/"))
        gt_box = map(float, annotation[1:5])
        gt_box = np.array(list(gt_box), dtype=np.int32)
        landmark = map(float, annotation[5:])
        landmark = np.array(list(landmark), dtype=np.float)
        img = cv2.imread(im_path)
        assert (img is not None)
        height, width, channel = img.shape
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
            resized_im = cv2.resize(cropped_im, (size, size),
                                    interpolation=cv2.INTER_LINEAR)
            offset_x1 = round((x1 - nx1) / float(bbox_size), 2)
            offset_y1 = round((y1 - ny1) / float(bbox_size), 2)
            offset_x2 = round((x2 - nx2) / float(bbox_size), 2)
            offset_y2 = round((y2 - ny2) / float(bbox_size), 2)
            offset_landmark = []
            for landmark_idx in range(len(landmark)):
                if landmark_idx % 2 == 0:
                    offset_landmark.append(round((landmark[landmark_idx] - nx1)
                                                 / float(bbox_size), 2))
                else:
                    offset_landmark.append(round((landmark[landmark_idx] - ny1)
                                                 / float(bbox_size), 2))
            # cal iou
            iou = utils.IoU(crop_box.astype(np.float),
                            np.expand_dims(gt_box.astype(np.float), 0))
            if iou > 0.65:
                # save_file = os.path.join(landmark_imgs_save_dir, "%s.jpg" % l_idx)
                # cv2.imwrite(save_file, resized_im)

                line = []
                # line.append(str(save_file))
                line.append(str(img_idx))

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

                img_list.append(resized_im)
                img_idx += 1

                l_idx += 1
    print("%d images done, landmark images: %d" % (idx, l_idx))
    f.close()

    np_path = os.path.join(data_dir, "48")
    if not os.path.exists(np_path):
        os.mkdir(np_path)
    print("%s images in img.npy" % len(img_list))
    np.save(os.path.join(np_path, 'img.npy'), img_list)
    print(img_idx)
    assert len(img_list) == img_idx, 'Images index error!'

    return save_landmark_anno



