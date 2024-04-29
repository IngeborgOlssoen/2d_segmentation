from data_prepros import train_data, val_data, test_data
from monai.data import DataLoader, Dataset
from monai.transforms import MapTransform
from monai.transforms import(
    Compose,
    ScaleIntensityRanged,
    RandRotate90d,
    RandFlipd,
    ToTensord,
    RandAffine,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandAdjustContrast,
    # RandElastic
    )


class RandAffineWithKeys(MapTransform):
    def __init__(self, keys, prob=0.5, translate_range=(15, 15), scale_range=(0.1, 0.1), rotate_range=(0.1)):
        super().__init__(keys)
        self.rand_affine = RandAffine(prob=prob, translate_range=translate_range, scale_range=scale_range, rotate_range=rotate_range)

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = self.rand_affine(data[key])
        return data

class RandGaussianNoiseWithKeys(MapTransform):
    def __init__(self, keys, prob=0.1, mean=0.0, std=0.1):
        super().__init__(keys)
        self.transform = RandGaussianNoise(prob=prob, mean=mean, std=std)

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = self.transform(data[key])
        return data

class RandGaussianSmoothWithKeys(MapTransform):
    def __init__(self, keys, prob=0.2, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5)):
        super().__init__(keys)
        self.transform = RandGaussianSmooth(prob=prob, sigma_x=sigma_x, sigma_y=sigma_y)

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = self.transform(data[key])
        return data

class RandAdjustContrastWithKeys(MapTransform):
    def __init__(self, keys, prob=0.3, gamma=(0.5, 1.5)):
        super().__init__(keys)
        self.transform = RandAdjustContrast(prob=prob, gamma=gamma)

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = self.transform(data[key])
        return data

def create_train_transforms():
    return Compose([
        ScaleIntensityRanged(
            keys=["img"],
            a_min=0, a_max=1000,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        RandAffineWithKeys(
            keys=["img", "seg"],
            prob=0.5,
            translate_range=(15, 15),
            scale_range=(0.1, 0.1),
            rotate_range=(0.1)
        ),
        # RandFlipd(
        #     keys=["img", "seg"],
        #     prob=0.5,
        #     spatial_axis=[1, 2]
        # ),
        # RandRotate90d(
        #     keys=["img", "seg"],
        #     prob=1.0,
        #     spatial_axes=[1, 2]
        # ),
        RandGaussianNoiseWithKeys(
            keys=["img"],
            prob=0.3,
            std=0.1
        ),
        # RandElastic(
        #     keys=["img", "seg"],
        #     prob=0.2,
        #     spacing=(30, 30),
        #     magnitude_range=(5, 6)
        # ),
        RandGaussianSmoothWithKeys(
            keys=["img"],
            prob=0.2,
            sigma_x=(0.5, 1.5),
            sigma_y=(0.5, 1.5)
        ),
        RandAdjustContrastWithKeys(
            keys=["img"],
            prob=0.3,
            gamma=(0.5, 1.5)
        ),

        #################prøve croppe bildet####

        ToTensord(keys=["img", "seg"])
    ])

def create_val_transforms():
    return Compose([
        ScaleIntensityRanged(
            keys=["img"],
            a_min=0, a_max=1000,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        ToTensord(keys=["img", "seg"])
    ])

def create_test_transforms():
    return Compose([
        ScaleIntensityRanged(
            keys=["img"],
            a_min=0, a_max=1000,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        ToTensord(keys=["img", "seg"])
    ])

# Example usage of the transform
train_transforms = create_train_transforms()
val_transforms = create_val_transforms()
test_transforms=create_test_transforms()

# # Transform the data using training and validation transforms
# transformed_train_data = [train_transforms(item) for item in train_data]
# transformed_val_data = [val_transforms(item) for item in val_data]

# print("Transformed Train Image Shape:", transformed_train_data["img"].shape)
# print("Transformed Val Image Shape:", transformed_val_data["seg"].shape)

    
    
train_dataset=Dataset(train_data, train_transforms)
val_dataset= Dataset(val_data, val_transforms)
test_dataset=Dataset(test_data,transform=test_transforms)


train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
val_loader=DataLoader(val_dataset,batch_size=2, shuffle=True, num_workers=4)
test_loader=DataLoader(test_dataset,batch_size=2,shuffle=True, num_workers=4)


# # Iterate through one epoch to see the debug output
# for i, data in enumerate(train_loader):
#     pass  # The debug statements will print during iteration

# for i, data in enumerate(val_loader):
#     pass  # Debug for validation data



#import torch
#from torch.utils.data import Dataset
#import numpy as np
#
#import torch
#from torch.utils.data import Dataset
#import numpy as np
#class CustomMedicalDataset(Dataset):
#    def __init__(self, data, transform=None):
#    #     """
#        # Args:
#        #     data (list of dicts): Data where each entry is a dict with keys 'img' and 'seg'.
#        #     transform (callable, optional): Transform to be applied on a sample.
#        # """
#        self.data = data
#        self.transform = transform
#
#    def __len__(self):
#        return len(self.data)
#    def __getitem__(self, idx):
#        sample = self.data[idx]
#
#        img = sample['img']
#        seg = sample['seg']
#
#        # Ensure data has at least a channel dimension
#        if img.ndim == 2:
#            img = img[np.newaxis, ...]  # Adding channel dimension
#        if seg.ndim == 2:
#            seg = seg[np.newaxis, ...]  # Adding channel dimension
#
#        sample['img'] = img
#        sample['seg'] = seg
#
#        if self.transform:
#            sample = self.transform(sample)
#
#        return sample
#
#    
#
#    def create_train_transforms():
#         return Compose([
#             ScaleIntensityRanged(keys=["img"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
#             RandAffineWithKeys(keys=["img", "seg"], prob=0.5, translate_range=(15, 15), scale_range=(0.1, 0.1), rotate_range=(0.1)),
#             RandFlipd(keys=["img", "seg"], prob=0.5, spatial_axis=[1, 2]),
#             RandRotate90d(keys=["img", "seg"], prob=1.0, spatial_axes=[1, 2]),
#             RandGaussianNoiseWithKeys(keys=["img"], prob=0.3, std=0.1),
#             RandGaussianSmoothWithKeys(keys=["img"], prob=0.2, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5)),
#             RandAdjustContrastWithKeys(keys=["img"], prob=0.3, gamma=(0.5, 1.5)),
#             ToTensord(keys=["img", "seg"])
#     ])


# Example of how to use the data loader
    
# try:
#     for i, data in enumerate(train_loader):
#         print(f"Batch {i} - Image shapes: {data['img'].shape}, Segmentation shapes: {data['seg'].shape}")
# except Exception as e:
#     print(f"Error processing batch {i}: {str(e)}")
#     raise
