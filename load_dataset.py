from __future__ import print_function
from scipy import misc
import os
import numpy as np
import sys

def load_test_data(phone, dped_dir, IMAGE_SIZE):

    test_directory_phone = dped_dir + str(phone) + '/test_data/patches/' + str(phone) + '/'
    test_directory_dslr = dped_dir + str(phone) + '/test_data/patches/canon/'

    NUM_TEST_IMAGES = len([name for name in os.listdir(test_directory_phone)
                           if os.path.isfile(os.path.join(test_directory_phone, name))])

    test_data = np.zeros((NUM_TEST_IMAGES, IMAGE_SIZE))
    test_answ = np.zeros((NUM_TEST_IMAGES, IMAGE_SIZE))

    for i in range(0, NUM_TEST_IMAGES):
        
        I = np.asarray(misc.imread(test_directory_phone + str(i) + '.jpg'))
        I = np.float16(np.reshape(I, [1, IMAGE_SIZE]))/255
        test_data[i, :] = I
        
        I = np.asarray(misc.imread(test_directory_dslr + str(i) + '.jpg'))
        I = np.float16(np.reshape(I, [1, IMAGE_SIZE]))/255
        test_answ[i, :] = I

        if i % 100 == 0:
            print(str(round(i * 100 / NUM_TEST_IMAGES)) + "% done", end="\r")

    return test_data, test_answ


def load_batch(phone, dped_dir, TRAIN_SIZE, IMAGE_SIZE):

    train_directory_phone = dped_dir + str(phone) + '/training_data/' + str(phone) + '/'
    train_directory_dslr = dped_dir + str(phone) + '/training_data/canon/'

    NUM_TRAINING_IMAGES = len([name for name in os.listdir(train_directory_phone)
                               if os.path.isfile(os.path.join(train_directory_phone, name))])

    # if TRAIN_SIZE == -1 then load all images

    if TRAIN_SIZE == -1:
        TRAIN_SIZE = NUM_TRAINING_IMAGES
        TRAIN_IMAGES = np.arange(0, TRAIN_SIZE)
    else:
        TRAIN_IMAGES = np.random.choice(np.arange(0, NUM_TRAINING_IMAGES), TRAIN_SIZE, replace=False)

    train_data = np.zeros((TRAIN_SIZE, IMAGE_SIZE))
    train_answ = np.zeros((TRAIN_SIZE, IMAGE_SIZE))

    i = 0
    for img in TRAIN_IMAGES:

        I = np.asarray(misc.imread(train_directory_phone + str(img) + '.jpg'))
        I = np.float16(np.reshape(I, [1, IMAGE_SIZE])) / 255
        train_data[i, :] = I

        I = np.asarray(misc.imread(train_directory_dslr + str(img) + '.jpg'))
        I = np.float16(np.reshape(I, [1, IMAGE_SIZE])) / 255
        train_answ[i, :] = I

        i += 1
        if i % 100 == 0:
            print(str(round(i * 100 / TRAIN_SIZE)) + "% done", end="\r")

    return train_data, train_answ






# Python
import os
import cv2
import random
# 3rd party
import numpy
import dill as pickle



def cv2_rotate(image, angle=15):
    height, width = image.shape[:2]    
    center = (width / 2, height / 2)   
    scale = 1                        
    M = cv2.getRotationMatrix2D(center, angle, scale)
    image_rotation = cv2.warpAffine(src=image, M=M, dsize=(width, height), borderValue=(0, 0, 0))
    return image_rotation

def cv2_show(image):
    cv2.imshow('crane', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



class ImageDataset():
    # numpy->torch.Tensor
    transform = lambda x: torch.from_numpy(x).permute(2, 0, 1).type(torch.FloatTensor).div(255)
    # torch.Tensor->numpy
    restore = lambda x: torch.clamp(x.detach().permute(0, 2, 3, 1), 0, 1).cpu().mul(255).numpy().astype('uint8')

    def __init__(self, images_list):
        self.images_list = images_list

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        return transform(cv2.imread(self.images_list[idx]))

    @staticmethod
    def make_paired_augment(low_quality, high_quality):
        # 以 0.6 的概率作数据增强
        if(random.random() > 1 - 0.9):
            # 待增强操作列表(如果是 Unet 的话, 其实这里可以加入一些旋转操作)
            all_states = ['crop', 'flip', 'rotate']
            # 打乱增强的顺序
            random.shuffle(all_states)
            for cur_state in all_states:
                if(cur_state == 'flip'):
                    # 0.5 概率水平翻转
                    if(random.random() > 0.5):
                        low_quality = cv2.flip(low_quality, 1)
                        high_quality = cv2.flip(high_quality, 1)
                        # print('水平翻转一次')
                elif(cur_state == 'crop'):
                    # 0.5 概率做裁剪
                    if(random.random() > 1 - 0.8):
                        H, W, _ = low_quality.shape
                        ratio = random.uniform(0.75, 0.95)
                        _H = int(H * ratio)
                        _W = int(W * ratio)
                        pos = (numpy.random.randint(0, H - _H), numpy.random.randint(0, W - _W))
                        low_quality = low_quality[pos[0]: pos[0] + _H, pos[1]: pos[1] + _W]
                        high_quality = high_quality[pos[0]: pos[0] + _H, pos[1]: pos[1] + _W]
                        # print('裁剪一次')
                elif(cur_state == 'rotate'):
                    # 0.2 概率旋转
                    if(random.random() > 1 - 0.1):
                        angle = random.randint(-15, 15)  
                        low_quality = cv2_rotate(low_quality, angle)
                        high_quality = cv2_rotate(high_quality, angle)
                        # print('旋转一次')
        return low_quality, high_quality

    @staticmethod
    def transform(x):
        H, W, C = x.shape
        return numpy.reshape(x, [1, H * W * C]) * 1. / 255

    @staticmethod
    def prepare_paired_images(input_path, label_path, augment, target_size):
        # 读取图像
        low_quality = cv2.imread(input_path)
        high_quality = cv2.imread(label_path)
        # 数据增强
        if(augment): 
            low_quality, high_quality = ImageDataset.make_paired_augment(low_quality, high_quality)
        # 分辨率要求
        if(target_size is not None): 
            low_quality = cv2.resize(low_quality, target_size)
            high_quality = cv2.resize(high_quality, target_size)
        # numpy->tensor
        return ImageDataset.transform(low_quality), ImageDataset.transform(high_quality), os.path.split(input_path)[-1]



class PairedImageDataset(ImageDataset):
    def __init__(self, images_list, augment=True, target_size=(256, 256)):
        super(PairedImageDataset, self).__init__(images_list)
        self.augment = augment
        self.target_size = target_size

    def __getitem__(self, idx):
        # 获取路径
        input_path, label_path = self.images_list[idx]
        low_quality, high_quality, image_name = ImageDataset.prepare_paired_images(input_path, label_path, self.augment, self.target_size)
        return low_quality, high_quality, image_name 
        




def get_images(opt):
    def update(data):
        return [(os.path.join(opt.dataset_dir, 'input', it), \
                os.path.join(opt.dataset_dir, 'expertC_gt', it)) for it in data]
    import dill as pickle
    with open('./FiveKSplitNew.pkl', 'rb') as file:
        data_split = pickle.load(file)
        train_valid_images_list = update(data_split['train_valid'])
        return train_valid_images_list[:4000], train_valid_images_list[4000:], update(data_split['test'])
            


if __name__ == '__main__':

    # 在这里验证下数据集是否正确?
    opt = lambda: None
    opt.dataset_dir = "D:/data/datasets/MIT-Adobe_FiveK/png"
    opt.A_dir = "input"
    opt.B_dir = "expertC_gt"
    opt.data_split = "./datasets/fiveK_split_new_unpaired.pkl"

    train_images_list, valid_images_list, test_images_list = get_images(opt)
    


    
    # pair === test
    test_dataset = PairedImageDataset(test_images_list, augment=False, target_size=None)



    from torch.utils.data import DataLoader    

    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)
    for test_batch, (A_image, B_image, image_name) in enumerate(test_dataloader, 1):
        # A_image_back = ImageDataset.restore(A_image)[0]
        # B_image_back = ImageDataset.restore(B_image)[0]
        # cv2_show(numpy.concatenate([A_image_back, B_image_back], axis=1))
        # if(test_batch == 5): break

        print(A_image.shape)
        break