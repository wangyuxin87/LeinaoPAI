import os
import numpy.random as npr
import numpy as np


def assemble_data(output_train, output_valid, anno_file_list=[]):

    if len(anno_file_list) == 0:
        return 0

    if os.path.exists(output_train):
        os.remove(output_train)
    if os.path.exists(output_valid):
        os.remove(output_valid)

    train_count = 0
    valid_count = 0
    for anno_file in anno_file_list:
        with open(anno_file, 'r') as f:
            anno_lines = f.readlines()

        base_num = 250000

        if len(anno_lines) > base_num * 3:
            idx_keep = npr.choice(len(anno_lines), size=base_num * 3, replace=True)
        elif len(anno_lines) > 100000:
            idx_keep = npr.choice(len(anno_lines), size=len(anno_lines), replace=True)
        else:
            idx_keep = np.arange(len(anno_lines))
        np.random.shuffle(idx_keep)
        train_keep = idx_keep[:int(0.9 * len(idx_keep))]
        valid_keep = idx_keep[int(0.9 * len(idx_keep)):]
        with open(output_train, 'a+') as f:
            for idx in train_keep:
                f.write(anno_lines[idx])
                train_count += 1
        with open(output_valid, 'a+') as f:
            for idx in valid_keep:
                f.write(anno_lines[idx])
                valid_count += 1
    return train_count, valid_count