# python test_model.py model=iphone_orig dped_dir=dped/ test_subset=full iteration=all resolution=orig use_gpu=true
import imageio
import os
import sys
import cv2
import numpy
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from models import resnet
import utils

tf.compat.v1.disable_v2_behavior()

# 规定网络输入, 为什么分辨率大? 因为后面有 resize, 低分辨率上采样在高频信号有损失
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE = 512, 512, 512 * 512 * 3

# GPU
use_gpu = True
config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}) if use_gpu == "false" else None

# 开始画推理图
x_ = tf.compat.v1.placeholder(tf.float32, [None, IMAGE_SIZE])
x_image = tf.reshape(x_, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
enhanced = resnet(x_image)

# 准备输入文件夹以及保存图片的文件夹
input_dir = './demo/input'
save_dir = "./demo/output"
os.makedirs(save_dir, exist_ok=True)
test_images_list = [os.path.join(input_dir, it) for it in os.listdir(input_dir)]

# 训练好的模型位置
pretrained_model = "./pretrained/FiveKNewSplit_iteration_168000.ckpt"

# 实际推理过程
with tf.compat.v1.Session(config=config) as sess:

    # 加载模型
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, pretrained_model)
    print('loaded model from {}'.format(pretrained_model))

    for cnt, image_path in enumerate(test_images_list, 1):
        # 读取图像
        origin_image = cv2.imread(image_path)
        H, W = origin_image.shape[:2]
        # 预处理
        image = cv2.resize(origin_image, (IMAGE_HEIGHT, IMAGE_WIDTH)) * 1. / 255
        image_crop_2d = numpy.reshape(image, [1, IMAGE_SIZE])
        # 经过网络增强
        enhanced_image = sess.run(enhanced, feed_dict={x_: image_crop_2d})
        # 后处理
        enhanced_image = (numpy.clip(enhanced_image, 0, 1) * 255).astype("uint8")
        enhanced_image = numpy.reshape(enhanced_image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        enhanced_image = cv2.resize(enhanced_image, (W, H))
        # 保存(只处理 png, 如果是其它格式另加参数)
        image_name = os.path.split(image_path)[-1];
        cv2.imwrite(os.path.join(save_dir, image_name), enhanced_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # 进度
        sys.stdout.write('\r{}/{}===>  saved to {}'.format(cnt, len(test_images_list), os.path.join(save_dir, image_name)))

    