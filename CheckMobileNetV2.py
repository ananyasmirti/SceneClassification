from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
import cv2


def load_image(img_path, show=False):

    #img = image.load_img(img_path, target_size=(256, 256))
    # (height, width, channels)
    img_tensor = image.img_to_array(img_path)  # img
    # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # imshow expects values in the range [0, 1]
    img_tensor /= 255.

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


if __name__ == "__main__":
    id1 = 0
    od = 0
    # load model
    model = load_model("D:/DRDO/placesMobilnet.h5")

    path = "D:/DRDO/hh"
    for image_path in os.listdir(path):
        # image path
        #img_path = 'D:/DRDO/testSetPlaces205_resize/00ffa154cfd6ddc5c3b483bf1c2976bb.jpg'
        #img_path = 'D:/Places2/train/outdoor/00000004.jpg'

        # load a single image
        input_path = os.path.join(path, image_path)
        new_image = imageio.imread(input_path)
        new_image = cv2.resize(new_image, (256, 256))
        new_image = load_image(new_image)

        # check prediction
        pred = model.predict_classes(new_image)

        if pred == 0:
            print("indoor")
            id1 += 1
        else:
            print("outdoor")
            od += 1
print("indoor:", id1)
print("outdoor:", od)
