import os
import data_prepare.assemble as assemble


def assemble_rnet_data(pnet_postive_file, pnet_neg_file, pnet_part_file):
    rnet_imglist_filename = './data'
    anno_list = []
    anno_list.append(pnet_postive_file)
    anno_list.append(pnet_part_file)
    anno_list.append(pnet_neg_file)

    imglist_train = os.path.join(rnet_imglist_filename, "train_anno_24.txt")
    imglist_valid = os.path.join(rnet_imglist_filename, "valid_anno_24.txt")

    train_count, valid_count = assemble.assemble_data(imglist_train, imglist_valid, anno_list)
    print("{} PNet train annotation result file path:{}".format(train_count, imglist_train))
    print("{} PNet train annotation result file path:{}".format(valid_count, imglist_valid))
