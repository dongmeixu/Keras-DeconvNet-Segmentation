from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.layers.convolutional import Conv2DTranspose
from keras.optimizers import Adam
from metrics import *
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import tifffile as tiff
import cv2

from collections import defaultdict

# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

FLAGS = tf.flags.FLAGS
# tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_integer("image_size", "160", "image size for training")
tf.flags.DEFINE_integer("image_channels", "3", "image channels for training")
tf.flags.DEFINE_integer("mask_channels", "9", "mask channels for output")

tf.flags.DEFINE_string("data_dir", "/media/files/xdm/ningxia-hn1/dataset/", "path to dataset")
tf.flags.DEFINE_string("model_dir", "DeconvNet/model/", "Path to vgg model mat")
tf.flags.DEFINE_string("npy_dir", "/home/xdm/deployed_projects/fcn_rnn/vgg16-rnn/npy/", "Path to npy-data")
tf.flags.DEFINE_string("mask_dir", "DeconvNet/mask/", "Path to mask-data")

windows_dir = r"F:\三百米裁切\hn1"
TWF_FILE = windows_dir + '\hn1.tfw'
DF = pd.read_csv(windows_dir + '\hn1_train_wkt.csv')
GS = pd.read_csv(windows_dir + '\hn1_grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

# TWF_FILE = FLAGS.data_dir + 'hn1.tfw'
# DF = pd.read_csv(FLAGS.data_dir + 'hn1_train_wkt.csv')
# GS = pd.read_csv(FLAGS.data_dir + 'hn1_grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)


def get_patches(img, msk, amt=10000, aug=True):
    is2 = int(1.0 * FLAGS.image_size)
    xm, ym = img.shape[0] - is2, img.shape[1] - is2

    x, y = [], []

    tr = [0.4, 0.1, 0.1, 0.15, 0.3, 0.95, 0.1, 0.05, 0.001]
    for i in range(amt):
        xc = random.randint(0, xm)
        yc = random.randint(0, ym)

        im = img[xc:xc + is2, yc:yc + is2]
        ms = msk[xc:xc + is2, yc:yc + is2]

        for j in range(FLAGS.mask_channels):
            sm = np.sum(ms[:, :, j])
            if 1.0 * sm / is2 ** 2 > tr[j]:
                if aug:
                    if random.uniform(0, 1) > 0.5:
                        im = im[::-1]
                        ms = ms[::-1]
                    if random.uniform(0, 1) > 0.5:
                        im = im[:, ::-1]
                        ms = ms[:, ::-1]

                x.append(im)
                y.append(ms)

    x, y = 2 * np.transpose(x, (0, 1, 2, 3)) - 1, np.transpose(y, (0, 1, 2, 3))
    return x, y


def calc_jacc(model):
    img = np.load(FLAGS.npy_dir + 'x_tmp_%d.npy' % FLAGS.mask_channels)
    msk = np.load(FLAGS.npy_dir + 'y_tmp_%d.npy' % FLAGS.mask_channels)

    prd = model.predict(img, batch_size=4)
    print("prd.shape, msk.shape: ", prd.shape, msk.shape)
    avg, trs = [], []

    for i in range(FLAGS.mask_channels):
        t_msk = msk[:, i, :, :]
        t_prd = prd[:, i, :, :]
        t_msk = t_msk.reshape(msk.shape[0] * msk.shape[2], msk.shape[3])
        t_prd = t_prd.reshape(msk.shape[0] * msk.shape[2], msk.shape[3])
        m, b_tr = 0, 0
        for j in range(10):
            tr = j / 10.0
            pred_binary_mask = t_prd > tr

            jk = jaccard_similarity_score(t_msk, pred_binary_mask)
            if jk > m:
                m = jk
                b_tr = tr
        print(i, m, b_tr)
        avg.append(m)
        trs.append(b_tr)

    score = sum(avg) / 10.0
    return score, trs


def get_net():
    inputs = Input((FLAGS.image_size, FLAGS.image_size, FLAGS.image_channels))
    conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(inputs)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1_2)

    conv2_1 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(pool1)
    conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(conv2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2_2)

    conv3_1 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(pool2)
    conv3_2 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(conv3_1)
    conv3_3 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(conv3_2)
    conv3_3 = BatchNormalization()(conv3_3)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3_3)

    conv4_1 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(pool3)
    conv4_2 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv4_1)
    conv4_3 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv4_2)
    conv4_3 = BatchNormalization()(conv4_3)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4_3)

    conv5_1 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(pool4)
    conv5_2 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv5_1)
    conv5_3 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv5_2)
    conv5_3 = BatchNormalization()(conv5_3)
    pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv5_3)

    fc6 = Conv2D(filters=4096, kernel_size=(5, 5), activation="relu")(pool5)
    fc7 = Conv2D(filters=4096, kernel_size=(1, 1), activation="relu")(fc6)

    deconv_fc6 = Conv2DTranspose(filters=512, kernel_size=(5, 5))(fc7)
    deconv_fc6 = BatchNormalization()(deconv_fc6)
    unpool5 = UpSampling2D(size=(2, 2))(deconv_fc6)

    deconv5_1 = Conv2DTranspose(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(unpool5)
    deconv5_2 = Conv2DTranspose(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(deconv5_1)
    deconv5_3 = Conv2DTranspose(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(deconv5_2)
    deconv5_3 = BatchNormalization()(deconv5_3)
    unpool4 = UpSampling2D(size=(2, 2))(deconv5_3)

    deconv4_1 = Conv2DTranspose(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(unpool4)
    deconv4_2 = Conv2DTranspose(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(deconv4_1)
    deconv4_3 = Conv2DTranspose(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(deconv4_2)
    deconv4_3 = BatchNormalization()(deconv4_3)
    unpool3 = UpSampling2D(size=(2, 2))(deconv4_3)

    deconv3_1 = Conv2DTranspose(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(unpool3)
    deconv3_2 = Conv2DTranspose(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(deconv3_1)
    deconv3_3 = Conv2DTranspose(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(deconv3_2)
    deconv3_3 = BatchNormalization()(deconv3_3)
    unpool2 = UpSampling2D(size=(2, 2))(deconv3_3)

    deconv2_1 = Conv2DTranspose(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(unpool2)
    deconv2_2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(deconv2_1)
    deconv2_2 = BatchNormalization()(deconv2_2)
    unpool1 = UpSampling2D(size=(2, 2))(deconv2_2)

    deconv1_1 = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(unpool1)
    deconv1_2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(deconv1_1)
    deconv1_2 = BatchNormalization()(deconv1_2)
    output = Conv2D(filters=FLAGS.mask_channels, kernel_size=(1, 1), activation='sigmoid')(deconv1_2)

    model = Model(input=inputs, output=output)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    return model


def train_net():
    print("start train net")
    # 读取训练以及验证数据
    '''linux'''
    x_val, y_val = np.load(FLAGS.npy_dir + 'x_tmp_%d.npy' % FLAGS.mask_channels), \
                   np.load(FLAGS.npy_dir + 'y_tmp_%d.npy' % FLAGS.mask_channels)
    img = np.load(FLAGS.npy_dir + 'x_trn_%d.npy' % FLAGS.mask_channels)
    msk = np.load(FLAGS.npy_dir + 'y_trn_%d.npy' % FLAGS.mask_channels)

    print("img.shape: ", img.shape)
    print("msk.shape: ", msk.shape)
    x_trn, y_trn = get_patches(img, msk)

    model = get_net()
    print(model.summary())
    # model.load_weights(FLAGS.model_dir + 'epoch_3_unet_9_jk0.8385')
    model_checkpoint = ModelCheckpoint(FLAGS.model_dir + 'unet_tmp.hdf5', monitor='loss', save_best_only=True)
    nb_epoch = 50
    for i in range(2):
        model.fit(x_trn, y_trn, batch_size=64, nb_epoch=nb_epoch, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint], validation_data=(x_val, y_val))
        del x_trn
        del y_trn
        x_trn, y_trn = get_patches(img, msk)
        score, trs = calc_jacc(model)
        print('val jk', score)
        model.save_weights(FLAGS.model_dir + 'epoch_' + str(i) + '_unet_9_jk%.4f' % score)

    return model


# model = train_net()
# calc_jacc(model)
print(get_net().summary())