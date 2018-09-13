import os
import data_prepare.assemble as assemble


def assemble_onet_data(onet_postive_file, onet_neg_file, onet_part_file, net_landmark_file):
    onet_imglist_filename = './data'
    anno_list = []

    anno_list.append(onet_postive_file)
    anno_list.append(onet_part_file)
    anno_list.append(onet_neg_file)
    anno_list.append(net_landmark_file)

    imglist_train = os.path.join(onet_imglist_filename, "train_anno_48.txt")
    imglist_valid = os.path.join(onet_imglist_filename, "valid_anno_48.txt")

    train_count, valid_count = assemble.assemble_data(imglist_train, imglist_valid, anno_list)
    print("{} PNet train annotation result file path:{}".format(train_count, imglist_train))
    print("{} PNet train annotation result file path:{}".format(valid_count, imglist_valid))
