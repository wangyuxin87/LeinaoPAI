import cv2
import time
import os


class DATA:
    def __init__(self, image_name, bboxes, landmarks):
        self.image_name = image_name
        self.bboxes = bboxes
        self.landmarks = landmarks


class W300(object):
    def __init__(self, path_to_image=None):
        self.root_dir = path_to_image

        self.file_to_image = list(filter(lambda x: os.path.splitext(x)[1] == '.png' or
                                                os.path.splitext(x)[1] == '.jpg', os.listdir(path_to_image)))
        self.file_to_label = list(filter(lambda x: os.path.splitext(x)[1] == '.pts', os.listdir(path_to_image)))

        # print(self.file_to_image)
        # print(self.file_to_label)
        # self.f = loadmat(file_to_label)
        # self.event_list = self.f['event_list']
        # self.file_list = self.f['file_list']
        # self.face_bbx_list = self.f['face_bbx_list']

    def next(self):
        for event_idx, path_of_image in enumerate(self.file_to_image):
            labels = list(filter(lambda x: os.path.splitext(path_of_image)[0] == os.path.splitext(x)[0],
                                 self.file_to_label))

            path_of_image = os.path.join(self.root_dir, path_of_image)
            bboxes = []
            for label in labels:
                path_of_label = os.path.join(self.root_dir, label)

                with open(path_of_label) as f:
                    lines = f.readlines()[3:71]

                x = []
                y = []
                landmarks = []
                for line in lines:
                    annotation = line.strip().split()
                    landmarks.append(int(float(annotation[0])))
                    landmarks.append(int(float(annotation[1])))
                    x.append(annotation[0])
                    y.append(annotation[1])
                x = [float(i) for i in x]
                y = [float(i) for i in y]
                bboxes.append([int(min(x)), int(min(y)), int(max(x)), int(max(y))])

            # from PIL import Image, ImageDraw
            # img = Image.open(path_of_image)
            # for idx in range(len(bboxes)):
            #     boxes = bboxes[idx]
            #     draw1 = ImageDraw.Draw(img)
            #     draw1.rectangle([boxes[0], boxes[1], boxes[2], boxes[3]])
            # img.show()

            # print(path_of_image)
            # print(bboxes)
            # print([landmarks])
            # if 'image_092' in path_of_image:
            #     print(path_of_image)
            yield DATA(path_of_image, bboxes, [landmarks])


# wider face original images path
# path_to_image = '300-W'
path_to_image = '300-W'

# target file path
target_file = './landmark_anno.txt'

wider = W300(path_to_image)

wider.next()


line_count = 0
box_count = 0
landmark_count = 0

print('start transforming....')
t = time.time()

with open(target_file, 'w+') as f:
    # press ctrl-C to stop the process
    for data in wider.next():
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


