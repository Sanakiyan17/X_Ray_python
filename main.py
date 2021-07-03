import os
from copy import copy
import imutils
import cv2
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from skimage import morphology, io, exposure, color, img_as_float, transform
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras import Model
from PIL import Image as cr_image, ImageOps
from tensorflow.python.keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


def loadDataGeneral(df, path, im_shape):
    X, y = [], []
    for i, item in df.iterrows():
        img = img_as_float(io.imread(path + item[0]))
        mask = io.imread(path + item[1])
        img = transform.resize(img, im_shape)
        img = exposure.equalize_hist(img)
        img = np.expand_dims(img, -1)
        mask = transform.resize(mask, im_shape)
        mask = exposure.equalize_hist(mask)
        # mask = np.expand_dims(mask, -1)
        X.append(img)
        y.append(mask)
    X = np.array(X)
    y = np.array(y)
    X -= X.mean()
    X /= X.std()

    print('### Dataset loaded')
    # print('Values in Y array:', y)
    print('\t{}\t{}'.format(X.shape, y.shape))
    print('\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), y.min(), y.max()))
    print('\tX.mean = {}, X.std = {}'.format(X.mean(), X.std()))
    return X, y


def IoU(y_true, y_pred):
    """Returns Intersection over Union score for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)


def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)


def masked(img, gt, mask, alpha=1):
    """Returns image with Humerus field outlined with red, predicted Humerus field
    filled with blue."""
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    boundary = np.subtract(morphology.dilation(gt, morphology.disk(3)), gt, dtype=np.float32)
    color_mask[mask == 1] = [0, 0, 1]
    color_mask[boundary == 1] = [1, 0, 0]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked


def build_UNet2D_4L(inp_shape, k_size=3):
    merge_axis = -1  # Feature maps are concatenated along last axis (for tf backend)
    data = Input(shape=inp_shape)
    conv1 = Convolution2D(filters=32, kernel_size=k_size, padding='same', activation='relu')(data)
    conv1 = Convolution2D(filters=32, kernel_size=k_size, padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(pool1)
    conv2 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(pool2)
    conv3 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(pool3)
    conv4 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(pool4)

    up1 = UpSampling2D(size=(2, 2))(conv5)
    conv6 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(up1)
    conv6 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(conv6)
    merged1 = concatenate([conv4, conv6], axis=merge_axis)
    conv6 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(merged1)

    up2 = UpSampling2D(size=(2, 2))(conv6)
    conv7 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(up2)
    conv7 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(conv7)
    merged2 = concatenate([conv3, conv7], axis=merge_axis)
    conv7 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(merged2)

    up3 = UpSampling2D(size=(2, 2))(conv7)
    conv8 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(up3)
    conv8 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(conv8)
    merged3 = concatenate([conv2, conv8], axis=merge_axis)
    conv8 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(merged3)

    up4 = UpSampling2D(size=(2, 2))(conv8)
    conv9 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(up4)
    conv9 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv9)
    merged4 = concatenate([conv1, conv9], axis=merge_axis)
    conv9 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(merged4)

    conv10 = Convolution2D(filters=1, kernel_size=k_size, padding='same', activation='sigmoid')(conv9)

    output = conv10
    model = Model(data, output)
    return model


def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Hello World')
    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    csv_path = "D:/Innovation2020/MedicalImaging/Prototype/patientXRays.csv"
    # Path to the folder with images. Images will be read from path + path_from_csv
    path = "D:/Innovation2020/MedicalImaging/Prototype/Images/"

    df = pd.read_csv(csv_path)

    # Specify which GPU(s) to use
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or 2, 3, etc. other than 0
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # On CPU/GPU placement
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)
    print("Output Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # print(df.head())
    # Load test data
    im_shape = (256, 256)
    X, y = loadDataGeneral(df, path, im_shape)

    # Set up a colormap:
    # use copy so that we do not mutate the global colormap instance
    palette = copy(plt.cm.twilight)
    palette.set_over('r', 1.0)
    palette.set_under('g', 1.0)
    palette.set_bad('b', 1.0)

    n_test = X.shape[0]
    inp_shape = X[0].shape
    print("Input Array:", inp_shape)

    UNet = build_UNet2D_4L(inp_shape, 3)
    UNet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Visualize model
    # plot_model(UNet, 'model.png', show_shapes=True)

    # train_gen = ImageDataGenerator(rotation_range=10,
    #                               width_shift_range=0.1,
    #                               height_shift_range=0.1,
    #                               rescale=1.,
    #                               zoom_range=0.2,
    #                               fill_mode='nearest',
    #                               cval=0)

    # For inference standard keras ImageGenerator can be used.
    test_gen = ImageDataGenerator(rescale=1.)

    ious = np.zeros(n_test)
    dices = np.zeros(n_test)

    gts, prs = [], []
    i = 0
    plt.figure(figsize=(10, 10))
    for xx, yy in test_gen.flow(X, y, batch_size=1):
        img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0, 1))
        pred = UNet.predict(xx)[..., 0].reshape(inp_shape[:2])
        mask = yy[..., 0].reshape(inp_shape[:2])

        gt = mask > 0.5
        pr = pred >= 0.5

        pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))

        io.imsave('output.png', masked(img, gt, pr, 1))

        gts.append(gt)
        prs.append(pr)
        ious[i] = IoU(gt, pr)
        dices[i] = Dice(gt, pr)
        print(df.iloc[i][0], ious[i], dices[i])

        if i < 4:
            plt.subplot(4, 4, 4 * i + 1)
            plt.title('Processed ' + df.iloc[i][0])
            plt.axis('off')
            plt.imshow(img, cmap='gray')

            plt.subplot(4, 4, 4 * i + 2)
            plt.title('IoU = {:.4f}'.format(ious[i]))
            plt.axis('off')
            plt.imshow(masked(img, gt, pr, 1), cmap='RdGy')

            plt.subplot(4, 4, 4 * i + 3)
            plt.title('Prediction')
            plt.axis('off')
            plt.imshow(pred, cmap="RdGy")

            pic_input = cr_image.fromarray(img)
            pic_predict = cr_image.fromarray(pred)

            io.imsave('interim_inp_img.png', img)
            io.imsave('pred_img.png', pred)

            # convert arrays to Images of grayscale
            inpimgA = cv2.imread('interim_inp_img.png', cv2.COLOR_BGR2GRAY)
            predB = cv2.imread('pred_img.png', cv2.COLOR_BGR2GRAY)

            (score, diff) = ssim(inpimgA, predB, full=True)
            diff = (diff * 255).astype("uint8")
            print("SSIM: {}".format(score))

            thresh = cv2.threshold(diff, 0, 255,
                                   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            plt.subplot(4, 4, 4 * i + 4)
            plt.title('Difference')
            plt.axis('off')
            plt.imshow(diff, cmap="Spectral")

            #plt.subplot(4, 4, 4 * i + 4)
            #plt.title('Threshold')
            #plt.axis('off')
            #plt.imshow(thresh, cmap="gray")

        i += 1
        if i == n_test:
            break

    print('Mean IoU:', ious.mean())
    print('Mean Dice:', dices.mean())
    plt.tight_layout()
    plt.savefig('results.png')
    plt.show()
    # exit(0)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
