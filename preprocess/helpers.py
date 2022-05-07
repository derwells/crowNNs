import os
import numpy as np

from deepforest import preprocess
from deepforest import utilities
from deepforest import main
from sklearn.model_selection import train_test_split
from config import *


def get_xml(img_fname):
    name = img_fname.split(".")[0]
    target_xml = XML_PATH + name + ".xml"

    if os.path.exists(target_xml):
        return target_xml
    else:
        print(
            "{} cannot be found!".format(target_xml)
        )


def imgs_to_xml(img_paths):
    xmls = []
    for img in img_paths:
        xml_path = get_xml(img)
        xmls.append(xml_path)

    return xmls


def train_val_split(img_paths):
    val_paths = np.random.choice(
        img_paths,
        int(len(img_paths)*0.25) # 25% validation
    )

    return val_paths


def write_to_csv(df, csv_path):
    df.to_csv(
        csv_path,
        mode='a',
        header=not(os.path.exists(csv_path)),
        index=False
    )


def get_train_images(dir):
    img_paths = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        img_paths.extend(filenames)

    return img_paths


def preprocess_image(img_fname):
    img_annotations = preprocess.split_raster(
        path_to_raster=os.path.join(TIF_DIR, img_fname),
        annotations_file=ANNOTATIONS_PATH,
        base_dir=CROP_DIR,

        # Tile settings
        patch_size=PATCH_SIZE,
        patch_overlap=PATCH_OVERLAP
    )

    img_paths = img_annotations.image_path.unique()

    val_paths = train_test_split(img_paths)

    val_annots = img_annotations.loc[
        img_annotations.image_path.isin(val_paths)
    ]

    train_annots = img_annotations.loc[
        ~img_annotations.image_path.isin(val_paths)
    ]

    # Write to CSVs
    write_to_csv(val_annots, VAL_ANNOTATIONS_PATH)
    write_to_csv(train_annots, TRAIN_ANNOTATIONS_PATH)


def compile_annotations(xml_paths):
    for xml in xml_paths:
        temp_annotations = utilities.xml_to_annotations(xml)
        write_to_csv(temp_annotations, ANNOTATIONS_PATH)


def remove_files(paths):
    for path in paths:
        if os.exists(path):
            os.remove(path)
