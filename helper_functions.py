from imgaug import augmenters as iaa
import numpy as np

def get_indices(y):

    initial = 0
    current = 0
    index = []
    index.append(current)
    for indices in range(len(y)):
        if y[current] == y[initial]:
            current += 1
        else:
            index.append(current)
            initial = current

    return index
    
def augmentations(img, label, size = 6, width = 32, height = 32, channels = 1):

    flipper = iaa.Fliplr(1.0) # Flip from left to right
    translaterH = iaa.Affine(translate_px = {"x": -10}) # Translate -10 pixels
    scaleV = iaa.Affine(scale = {"y": 1.5}) # Stretch 1.5 times in y_direction
    rotate = iaa.Affine(rotate = (-20,20)) # Rotate -20 degrees to 20 degrees
    shear = iaa.Affine(shear = (-16, 16)) # Shear -16 degrees to 16 degrees
    brightness = iaa.Add(-20) # For brightness

    augmented_images = np.zeros((size, width, height, channels))
    labels = np.zeros(size)

    augmented_images[0] = flipper.augment_image(img)
    augmented_images[1] = translaterH.augment_image(img)
    augmented_images[2] = scaleV.augment_image(img)
    augmented_images[3] = rotate.augment_image(img)
    augmented_images[4] = shear.augment_image(img)
    augmented_images[5] = brightness.augment_image(img)

    for i in range(size):
        labels[i] = label

    return augmented_images, labels
