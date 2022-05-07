import os
import time
import preprocess.helpers

from config import *


def do_preprocessing():
    # Remove previously-generated files
    preprocess.helpers.remove_paths([
        ANNOTATIONS_PATH,
        TRAIN_ANNOTATIONS_PATH,
        VAL_ANNOTATIONS_PATH,
        CROP_DIR
    ])

    train_imgs = preprocess.helpers.get_train_images(TIF_DIR)
    train_xmls = preprocess.helpers.imgs_to_xml(train_imgs)

    # Build compiled annotations
    preprocess.helpers.compile_annotations(train_xmls)

    # Build cropped images folder + annotations in CROP_DIR
    start_time = time.time()

    for img in train_imgs:
        preprocess.helpers.preprocess_image(img)

    print(f"--- Preprocessing: {(time.time() - start_time):.2f} seconds ---")


if __name__ == "__main__":
    do_preprocessing()
