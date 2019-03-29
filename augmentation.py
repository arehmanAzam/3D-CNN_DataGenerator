import cv2
import random
import numpy as np
import tensorflow as tf
from scipy import ndimage

# img = cv2.imread("/home/cvml2/ActivityData/train/bagdrop/(5).jpg")
class augmentation():

    def apply_brightness_contrast(input_img, brightness, contrast):

        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()

        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

    def central_scale_images(X_imgs, scales):
        # Various settings needed for Tensorflow operation
        boxes = np.zeros((len(scales), 4), dtype=np.float32)
        for index, scale in enumerate(scales):
            x1 = y1 = 0.5 - 0.5 * scale  # To scale centrally
            x2 = y2 = 0.5 + 0.5 * scale
            boxes[index] = np.array([y1, x1, y2, x2], dtype=np.float32)
        box_ind = np.zeros((len(scales)), dtype=np.int32)
        crop_size = np.array([400, 256], dtype=np.int32)

        X_scale_data = []
        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, shape=(1, 400, 256, 3))
        # Define Tensorflow operation for all scales but only one base image at a time
        tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for img_data in X_imgs[0]:
                batch_img = np.expand_dims(img_data, axis=0)
                scaled_imgs = sess.run(tf_img, feed_dict={X: batch_img})
                X_scale_data.extend(scaled_imgs)

        X_scale_data = np.array(X_scale_data, dtype=np.uint8)

        return X_scale_data


    def flip(image):
        # print("augment flip")
        flipped_img = np.fliplr(image)
        return flipped_img

    def scaleImage(img):
        # cv2.imshow('old', img)
        y = 50
        x = 50
        h = 400
        w = 200
        crop = img[y:y + h, x:x + w]
        crop = cv2.resize(crop, (256, 400))
        return crop

    def rotate(img):
        # rotation angle in degree
        rotated = ndimage.rotate(img, -20)
        rotated = cv2.resize(rotated, (256, 400))
        return rotated

        #
    # i=0
    # while(True):
    #     id = random.randint(1,5)
    #
    #     print(id)
    #
    #
    #     if id == 1:
    #         # Call Brightness
    #         brightness(1)
    #         print("call Brightness")
    #
    #     if id == 2:
    #         # Call Scaling
    #         scaling(1)
    #         print("call Scaling")
    #
    #     if id == 3:
    #         # Call ZoomInOut
    #         zoomInOut(1)
    #         print("call ZoomInOut")
    #
    #     if id == 4:
    #         # Call Flip
    #         flip(1)
    #         print("call flip")