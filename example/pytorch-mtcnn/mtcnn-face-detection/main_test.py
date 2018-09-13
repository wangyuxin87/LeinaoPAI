import cv2
import infers.vision as vision
from infers.detect import create_mtcnn_net, MtcnnDetector


if __name__ == '__main__':
    pnet, rnet, onet = create_mtcnn_net(
        p_model_path="./experiments/mtcnn_exp_p0/checkpoints/checkpoint.pth.tar",
        r_model_path="./experiments/mtcnn_exp_r0/checkpoints/checkpoint.pth.tar",
        o_model_path="./experiments/mtcnn_exp_o0/checkpoints/model_best.pth.tar",
        use_cuda=True)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=12)

    img = cv2.imread("./demo.png")
    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxs, landmarks = mtcnn_detector.detect_face(img_bg)
    vision.vis_face(img_bg, bboxs, landmarks)
