import os

import pandas as pd
import albumentations
from sklearn import model_selection
from wtfml.data_loaders.image.classification import ClassificationDataset

from al.helpers.constants import DATA_ROOT
from al.dataset.active_dataset import ActiveDataset

DATA_PATH = os.getenv('TO_DATA_PATH')


def get_train_val_datasets():
    df = pd.read_csv(
        f"{DATA_ROOT}/siim-isic-melanoma-classification/train.csv")
    df = df.sample(frac=1).reset_index(drop=True)
    df_train, df_valid = model_selection.train_test_split(
        df, test_size=0.3, random_state=42)

    training_data_path = f"{DATA_PATH}/siic-isic-224x224-images/train/"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True),
            albumentations.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
            albumentations.Flip(p=0.5)
        ]
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )

    train_images = df_train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path, i + ".png")
                    for i in train_images]
    train_targets = df_train.target.values

    valid_images = df_valid.image_name.values.tolist()
    valid_images = [os.path.join(training_data_path, i + ".png")
                    for i in valid_images]
    valid_targets = df_valid.target.values

    train_dataset = ClassificationDataset(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_aug,
    )

    valid_dataset = ClassificationDataset(
        image_paths=valid_images,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug,
    )

    return train_dataset, valid_dataset


class MelanomaDataset(ActiveDataset):

    def __init__(self, train_dataset, val_dataset, n_init=100):
        self.dataset = train_dataset
        self.val_dataset = val_dataset
        super().__init__(train_dataset, n_init=n_init)


def get_dataset_object(n_init):
    train_ds, val_ds = get_train_val_datasets()
    print('Raw train size :', len(train_ds), 'validation size :', len(val_ds))
    return MelanomaDataset(train_ds, val_ds, n_init=n_init)


if __name__ == '__main__':
    train_ds, val_ds = get_train_val_datasets()
    print('Train size :', len(train_ds), 'validation size :', len(val_ds))
    dataset = MelanomaDataset(train_ds, val_ds)
    print('Labeled size :', len(dataset.get_labeled()))
    print('Unlabeled size :', len(dataset.get_unlabeled()))
    print('Validation size :', len(dataset.get_validation_dataset()))
