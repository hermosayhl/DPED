# python train_model.py model={iphone,sony,blackberry} dped_dir=dped vgg_dir=vgg_pretrained/imagenet-vgg-verydeep-19.mat

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import imageio
import numpy as np
import sys

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)

tf.compat.v1.disable_v2_behavior()



from load_dataset import load_test_data, load_batch, get_images, PairedImageDataset
from torch.utils.data import DataLoader   
opt = lambda: None
opt.dataset_dir = "./datasets/MIT-Adobe_FiveK"
opt.A_dir = "input"
opt.B_dir = "expertC_gt"
# 读取数据
train_images_list, valid_images_list, test_images_list = get_images(opt)
# pair === train
train_dataset = PairedImageDataset(train_images_list, augment=True, target_size=(100, 100))
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
# pair === valid
valid_dataset = PairedImageDataset(valid_images_list, augment=False, target_size=(100, 100))
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=8)
# pair === test
test_dataset = PairedImageDataset(test_images_list, augment=False, target_size=(100, 100))
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)





from ssim import MultiScaleSSIM
import models
import utils
import vgg

# defining size of the training image patches

PATCH_WIDTH = 100
PATCH_HEIGHT = 100
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3

# processing command arguments

phone, batch_size, train_size, learning_rate, num_train_iters, \
w_content, w_color, w_texture, w_tv, \
dped_dir, vgg_dir, eval_step = utils.process_command_args(sys.argv)

np.random.seed(0)

# defining system architecture

with tf.Graph().as_default(), tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
    
    # placeholders for training data

    phone_ = tf.compat.v1.placeholder(tf.float32, [None, PATCH_SIZE])
    phone_image = tf.reshape(phone_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])

    dslr_ = tf.compat.v1.placeholder(tf.float32, [None, PATCH_SIZE])
    dslr_image = tf.reshape(dslr_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])

    adv_ = tf.compat.v1.placeholder(tf.float32, [None, 1])

    # get processed enhanced image

    enhanced = models.resnet(phone_image)

    # transform both dslr and enhanced images to grayscale

    enhanced_gray = tf.reshape(tf.image.rgb_to_grayscale(enhanced), [-1, PATCH_WIDTH * PATCH_HEIGHT])
    dslr_gray = tf.reshape(tf.image.rgb_to_grayscale(dslr_image),[-1, PATCH_WIDTH * PATCH_HEIGHT])

    # push randomly the enhanced or dslr image to an adversarial CNN-discriminator

    adversarial_ = tf.multiply(enhanced_gray, 1 - adv_) + tf.multiply(dslr_gray, adv_)
    adversarial_image = tf.reshape(adversarial_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 1])

    discrim_predictions = models.adversarial(adversarial_image)

    # losses
    # 1) texture (adversarial) loss

    discrim_target = tf.concat([adv_, 1 - adv_], 1)

    loss_discrim = -tf.reduce_sum(discrim_target * tf.compat.v1.log(tf.clip_by_value(discrim_predictions, 1e-10, 1.0)))
    loss_texture = -loss_discrim

    correct_predictions = tf.equal(tf.argmax(discrim_predictions, 1), tf.argmax(discrim_target, 1))
    discim_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # 2) content loss

    CONTENT_LAYER = 'relu5_4'

    enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(enhanced * 255))
    dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(dslr_image * 255))

    content_size = utils._tensor_size(dslr_vgg[CONTENT_LAYER]) * batch_size
    loss_content = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER] - dslr_vgg[CONTENT_LAYER]) / content_size

    # 3) color loss

    enhanced_blur = utils.blur(enhanced)
    dslr_blur = utils.blur(dslr_image)

    loss_color = tf.reduce_sum(tf.pow(dslr_blur - enhanced_blur, 2))/(2 * batch_size)

    # 4) total variation loss

    batch_shape = (batch_size, PATCH_WIDTH, PATCH_HEIGHT, 3)
    tv_y_size = utils._tensor_size(enhanced[:,1:,:,:])
    tv_x_size = utils._tensor_size(enhanced[:,:,1:,:])
    y_tv = tf.nn.l2_loss(enhanced[:,1:,:,:] - enhanced[:,:batch_shape[1]-1,:,:])
    x_tv = tf.nn.l2_loss(enhanced[:,:,1:,:] - enhanced[:,:,:batch_shape[2]-1,:])
    loss_tv = 2 * (x_tv/tv_x_size + y_tv/tv_y_size) / batch_size

    # final loss

    loss_generator = w_content * loss_content + w_texture * loss_texture + w_color * loss_color + w_tv * loss_tv

    # psnr loss

    enhanced_flat = tf.reshape(enhanced, [-1, PATCH_SIZE])

    loss_mse = tf.reduce_sum(tf.pow(dslr_ - enhanced_flat, 2))/(PATCH_SIZE * batch_size)
    loss_psnr = 20 * utils.log10(1.0 / tf.sqrt(loss_mse))

    # optimize parameters of image enhancement (generator) and discriminator networks

    generator_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith("generator")]
    discriminator_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith("discriminator")]

    train_step_gen = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_generator, var_list=generator_vars)
    train_step_disc = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_discrim, var_list=discriminator_vars)

    saver = tf.compat.v1.train.Saver(var_list=generator_vars, max_to_keep=100)

    print('Initializing variables')
    sess.run(tf.compat.v1.global_variables_initializer())

    # loading training and test data

    print('Training network')

    train_loss_gen = np.zeros((1, 6))
    train_acc_discrim = 0.0
    all_zeros = np.reshape(np.zeros((batch_size, 1)), [batch_size, 1])

    logs = open('models/' + phone + '.txt', "w+")
    logs.close()


    def inf_gen(data_loader):
        while True:
            for inp, gt, name in data_loader:
                yield inp.squeeze(1).numpy(), gt.squeeze(1).numpy(), name

    # 训练集有多少个
    train_batch_nums = len(train_dataloader)

    # 无限数据生成器
    train_data_gen = inf_gen(train_dataloader)


    import sys
    import cv2
    import copy
    for train_iters in range(1, 400000):

        low_quality, high_quality, image_names = train_data_gen.__next__()

        # train generator
        # print('low_quality  ', low_quality.shape)
        # print('high_quality  ', high_quality.shape)
        # print('all_zeros  ', all_zeros.shape)

        # idx_train = np.random.randint(0, train_size, batch_size)

        # phone_images = train_data[idx_train]
        # dslr_images = train_answ[idx_train]

        [enhanced_result, loss_temp, temp, losses] = sess.run([enhanced, loss_generator, train_step_gen, \
                [loss_generator, loss_content, loss_color, loss_texture, loss_tv, loss_psnr]],
                                        feed_dict={phone_: low_quality, dslr_: high_quality, adv_: all_zeros})
        train_loss_gen += np.asarray(losses)


        if(train_iters % 2000 == 0):

            sample_input = np.reshape(copy.deepcopy(low_quality[0]), [PATCH_HEIGHT, PATCH_WIDTH, 3])
            sample_label = np.reshape(copy.deepcopy(high_quality[0]), [PATCH_HEIGHT, PATCH_WIDTH, 3])
            sample_output = np.reshape(copy.deepcopy(enhanced_result[0]), [PATCH_HEIGHT, PATCH_WIDTH, 3])
            sample_image = np.concatenate([sample_input, sample_label, sample_output], axis=1)
            # print('sample_image  ', sample_image.shape, sample_image.dtype, type(sample_image))
            save_path = 'visual_results/' + 'iteration_' + str(train_iters) + '.png'
            # imageio.imwrite(save_path, )
            cv2.imwrite(save_path, (np.clip(sample_image, 0, 1) * 255).astype('uint8'))
            print('\nvisual results are written into {}'.format(save_path))

        # train discriminator

        idx_train = np.random.randint(0, train_size, batch_size)

        # generate image swaps (dslr or enhanced) for discriminator
        swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])

        # 再次获取数据
        low_quality, high_quality, image_names = train_data_gen.__next__()


        [accuracy_temp, temp] = sess.run([discim_accuracy, train_step_disc],
                                        feed_dict={phone_: low_quality, dslr_: high_quality, adv_: swaps})
        train_acc_discrim += accuracy_temp / eval_step

        sys.stdout.write('\rTrain===>{}/{}  [PSNR {:.3f}]'.format(train_iters, 1000000, 
            train_loss_gen[0][5] / train_iters))

        # if train_iters % eval_step == 0:
        if train_iters % 4000 == 0:

            # test generator and discriminator CNNs

            test_losses_gen = np.zeros((1, 6))
            test_accuracy_disc = 0.0
            loss_ssim = 0.0

            num_test_batches = len(valid_dataloader)

            for test_batch, (low_quality, high_quality, image_names) in enumerate(valid_dataloader, 1):
                # tensor -> numpy
                low_quality = low_quality.squeeze(1).numpy()
                high_quality = high_quality.squeeze(1).numpy()

                # print('\nlow_quality  ', low_quality.shape)
                # print('high_quality  ', high_quality.shape)

                # be = j * batch_size
                # en = (j+1) * batch_size

                cur_batch_size = len(low_quality)

                swaps = np.reshape(np.random.randint(0, 2, cur_batch_size), [cur_batch_size, 1])

                # phone_images = test_data[be:en]
                # dslr_images = test_answ[be:en]

                [enhanced_crops, accuracy_disc, losses] = sess.run([enhanced, discim_accuracy, \
                                [loss_generator, loss_content, loss_color, loss_texture, loss_tv, loss_psnr]], \
                                feed_dict={phone_: low_quality, dslr_: high_quality, adv_: swaps})

                test_losses_gen += np.asarray(losses)
                test_accuracy_disc += accuracy_disc

                loss_ssim += MultiScaleSSIM(np.reshape(high_quality * 255, [cur_batch_size, PATCH_HEIGHT, PATCH_WIDTH, 3]),
                                                    enhanced_crops * 255)

                sys.stdout.write('\rValid===>{}/{} [PSNR {:.3f}]'.format(
                    test_batch, num_test_batches, test_losses_gen[0][5] / test_batch))
                # break

            logs_disc = "step %d, %s | discriminator accuracy | train: %.4g, test: %.4g" % \
                  (train_iters, phone, train_acc_discrim, test_accuracy_disc / test_batch)

            logs_gen = "generator losses | test: %.4g | content: %.4g, color: %.4g, texture: %.4g, tv: %.4g | psnr: %.4g, ms-ssim: %.4g\n" % \
                  (test_losses_gen[0][0] / test_batch, test_losses_gen[0][1] / test_batch, test_losses_gen[0][2] / test_batch,
                   test_losses_gen[0][3] / test_batch, test_losses_gen[0][4] / test_batch, test_losses_gen[0][5] / test_batch, loss_ssim / test_batch)

            print(logs_disc)
            print(logs_gen)

            # save the results to log file

            logs = open('models/' + phone + '.txt', "a")
            logs.write(logs_disc)
            logs.write('\n')
            logs.write(logs_gen)
            logs.write('\n')
            logs.close()

            # save visual results for several test image crops

            # enhanced_crops = sess.run(enhanced, feed_dict={phone_: test_crops, dslr_: dslr_images, adv_: all_zeros})

            # idx = 0
            # for crop in enhanced_crops:
            #     before_after = np.hstack((np.reshape(test_crops[idx], [PATCH_HEIGHT, PATCH_WIDTH, 3]), crop))
            #     imageio.imwrite('results/' + str(phone)+ "_" + str(idx) + '_iteration_' + str(i) + '.jpg', before_after)
            #     idx += 1

            train_loss_gen = np.zeros((1, 6))
            train_acc_discrim = 0.0

            # save the model that corresponds to the current iteration

            saver.save(sess, 'models/FiveKNewSplit' + '_iteration_' + str(train_iters) + '.ckpt', write_meta_graph=False)

            # reload a different batch of training data

            print('保存成功!')
            
