import cv2
import time
import os


class DATA:
    def __init__(self, image_name, bboxes, landmarks):
        self.image_name = image_name
        self.bboxes = bboxes
        self.landmarks = landmarks


class CelebA(object):
    def __init__(self, path_to_image, path_to_anno):
        self.root_dir = path_to_image

        self.root_dir = path_to_image
        self.file_to_bbox = os.path.join(path_to_anno, 'list_bbox_celeba.txt')
        self.file_to_landmark = os.path.join(path_to_anno, 'list_landmarks_celeba.txt')

        # print(self.file_to_image)
        # print(self.file_to_label)
        # self.f = loadmat(file_to_label)
        # self.event_list = self.f['event_list']
        # self.file_list = self.f['file_list']
        # self.face_bbx_list = self.f['face_bbx_list']

    def next(self):
        with open(self.file_to_bbox) as f:
            bboxes_lines = f.readlines()[2:]

        with open(self.file_to_landmark) as f:
            landmark_lines = f.readlines()[2:]

        assert len(bboxes_lines) == len(landmark_lines), 'Annotation length error!'

        for idx in range(len(bboxes_lines)):
        # for idx in range(5):
            bboxes_annotation = bboxes_lines[idx].strip().split()
            landmark_annotation = landmark_lines[idx].strip().split()
            assert bboxes_annotation[0] == landmark_annotation[0], 'Label error!'

            path_of_image = os.path.join(self.root_dir, bboxes_annotation[0])

            bboxes = []
            landmarks = []

            for bb in bboxes_annotation[1:]:
                bboxes.append(int(bb))
            bboxes = [int(i) for i in bboxes]
            bboxes[2] += bboxes[0]
            bboxes[3] += bboxes[1]
            bboxes = [bboxes]

            for ll in landmark_annotation[1:]:
                landmarks.append(int(ll))
            landmarks = [landmarks]
            print(landmarks)

            # from PIL import Image, ImageDraw
            # img = Image.open('../'+path_of_image)
            # for idx in range(len(bboxes)):
            #     boxes = bboxes[idx]
            #     draw1 = ImageDraw.Draw(img)
            #     draw1.rectangle([boxes[0], boxes[1], boxes[2], boxes[3]])
            #
            #     land = landmarks[idx]
            #     for dl in range(0, 10, 2):
            #         draw1.ellipse([land[dl]-5, land[dl+1]-5, land[dl]+5, land[dl+1]+5], fill='red')
            # img.show()

            yield DATA(path_of_image, bboxes, landmarks)


# wider face original images path
# path_to_image = '300-W'
path_to_image = 'CelebA/Img/img_celeba'
path_to_anno = '../CelebA/Anno'

# target file path
target_file = './landmark_anno.txt'

celeba = CelebA(path_to_image, path_to_anno)

celeba.next()


line_count = 0
box_count = 0
landmark_count = 0

print('start transforming....')
t = time.time()

with open(target_file, 'w+') as f:
    # press ctrl-C to stop the process
    for data in celeba.next():
        line = []
        line.append(str(data.image_name))
        line_count += 1
        for i, box in enumerate(data.bboxes):
            box_count += 1
            for j, bvalue in enumerate(box):
                line.append(str(bvalue))
        for i, landmark in enumerate(data.landmarks):
            landmark_count += 1
            for j, lvalue in enumerate(landmark):
                line.append(str(lvalue))
        line.append('\n')

        line_str = ' '.join(line)
        f.write(line_str)

st = time.time()-t
print('end transforming')

print('spend time:%ld'%st)
print('total line(images):%d'%line_count)
print('total boxes(faces):%d'%box_count)


