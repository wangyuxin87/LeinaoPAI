import argparse
from utils.config import *


def main():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()

    # parse the config json file
    config = process_config(args.config)
    if config.net == 'PNet':
        from data_prepare.gen_Pnet_train_data import gen_pnet_data
        from data_prepare.assemble_pnet_imglist import assemble_pnet_data

        post_file, neg_file, part_file = gen_pnet_data()
        assemble_pnet_data(post_file, neg_file, part_file)

    if config.net == 'RNet':
        from data_prepare.gen_Rnet_train_data import gen_rnet_data
        from data_prepare.assemble_rnet_imglist import assemble_rnet_data

        pnet_checkpoint = "./experiments/mtcnn_exp_p0/checkpoints/checkpoint.pth.tar"
        det_boxs_file = './data/prepare/detections_pnet.pkl'
        if not os.path.exists(det_boxs_file):
            post_file, neg_file, part_file = gen_rnet_data(
                pnet_checkpoint, use_cuda=config.cuda)

        else:
            from data_prepare.gen_Rnet_train_data import gen_rnet_sample_data

            data_dir = '/userhome/mtcnn_dataset/'
            anno_file = './data_prepare/WIDER_ANNO/wider_origin_anno.txt'
            prefix_path = '/gdata/WIDER/WIDER_train/images'
            post_file, neg_file, part_file = gen_rnet_sample_data(data_dir,
                                                                  anno_file,
                                                                  det_boxs_file,
                                                                  prefix_path)
        assemble_rnet_data(post_file, neg_file, part_file)

    if config.net == 'ONet':
        from data_prepare.gen_Onet_train_data import gen_onet_data
        from data_prepare.gen_CelebA_landmark_48 import gen_data
        from data_prepare.assemble_onet_imglist import assemble_onet_data

        pnet_checkpoint = "./experiments/mtcnn_exp_p0/checkpoints/checkpoint.pth.tar"
        rnet_checkpoint = "./experiments/mtcnn_exp_r0/checkpoints/checkpoint.pth.tar"
        det_boxs_file = './data/prepare/detections_rnet.pkl'
        if not os.path.exists(det_boxs_file):
            post_file, neg_file, part_file, img_list, img_idx = gen_onet_data(pnet_checkpoint,
                                                                              rnet_checkpoint,
                                                                              use_cuda=config.cuda)
        else:
            from data_prepare.gen_Onet_train_data import gen_onet_sample_data

            data_dir = '/userhome/mtcnn_dataset/'
            anno_file = './data_prepare/WIDER_ANNO/wider_origin_anno.txt'
            prefix_path = '/gdata/WIDER/WIDER_train/images'
            post_file, neg_file, part_file, img_list, img_idx = gen_onet_sample_data(data_dir,
                                                                                     anno_file,
                                                                                     det_boxs_file,
                                                                                     prefix_path)
        net_landmark_file = gen_data(img_list, img_idx)
        assemble_onet_data(post_file, neg_file, part_file, net_landmark_file)


if __name__ == '__main__':
    main()
