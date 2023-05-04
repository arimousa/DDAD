import os
from glob import glob
from pathlib import Path
import shutil
import numpy as np
import csv
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR10


class CIFAR10_dataset():
    def __init__(self, config):
        self.splits = ['train', 'test']
        self.drop_last_batch = {'train': True, 'test': False}
        self.shuffle = {'train': True, 'test': False}
        self.batch_size = config.data.batch_size
        self.category = config.data.category
        self.manualseed = config.data.manualseed
        self.num_workers = config.model.num_workers



        self.transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                transforms.Lambda(lambda t: (t * 2) - 1)
            ]
        )

    def __getitem__(self):
        

        classes = {
            'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
            'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
        }

        dataset = {}
        dataset['train'] = CIFAR10(root='./datasets/CIFAR10', train=True, download=True, transform=self.transform)
        dataset['test'] = CIFAR10(root='./datasets/CIFAR10', train=False, download=True, transform=self.transform)

        dataset['train'].data, dataset['train'].targets, \
        dataset['test'].data, dataset['test'].targets = get_cifar_anomaly_dataset(
            trn_img=dataset['train'].data,
            trn_lbl=dataset['train'].targets,
            tst_img=dataset['test'].data,
            tst_lbl=dataset['test'].targets,
            abn_cls_idx=classes[self.category],
            manualseed=self.manualseed
        )

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                        batch_size=self.batch_size,
                                                        shuffle=self.shuffle[x],
                                                        num_workers=int(self.num_workers),
                                                        drop_last=self.drop_last_batch[x],
                                                        worker_init_fn=(None if self.manualseed == -1
                                                        else lambda x: np.random.seed(self.manualseed)))
                        for x in self.splits}


        return dataloader



class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, config, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),  
                # transforms.CenterCrop(224), 
                transforms.ToTensor(), # Scales data into [0,1] 
                transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
            ]
        )
        self.config = config
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),
                # transforms.CenterCrop(224),
                transforms.ToTensor(), # Scales data into [0,1] 
            ]
        )
        if is_train:
            if category:
                self.image_files = glob(
                    os.path.join(root, category, "train", "good", "*.png")
                )
            else:
                self.image_files = glob(
                    os.path.join(root, "train", "good", "*.png")
                )
        else:
            if category:
                self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            else:
                self.image_files = glob(os.path.join(root, "test", "*", "*.png"))
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.image_transform(image)
        if(image.shape[0] == 1):
            image = image.expand(3, self.config.data.image_size, self.config.data.image_size)
        if self.is_train:
            label = 'good'
            return image, label
        else:
            if self.config.data.mask:
                if os.path.dirname(image_file).endswith("good"):
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 'good'
                else :
                    if self.config.data.name == 'MVTec':
                        target = Image.open(
                            image_file.replace("/test/", "/ground_truth/").replace(
                                ".png", "_mask.png"
                            )
                        )
                    else:
                        target = Image.open(
                            image_file.replace("/test/", "/ground_truth/"))
                    target = self.mask_transform(target)
                    # target = F.interpolate(target.unsqueeze(1) , size = int(self.config.data.image_size), mode="bilinear").squeeze(1)
                    label = 'defective'
            else:
                if os.path.dirname(image_file).endswith("good"):
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 'good'
                else :
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 'defective'
                
            return image, target, label

    def __len__(self):
        return len(self.image_files)




def load_data(dataset_name='cifar10',normal_class=0,batch_size= 32):


    img_transform = transforms.Compose([
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    os.makedirs("./Dataset/CIFAR10/train", exist_ok=True)
    dataset = CIFAR10('./Dataset/CIFAR10/train', train=True, download=True, transform=img_transform)
    print("Cifar10 DataLoader Called...")
    print("All Train Data: ", dataset.data.shape)
    dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
    dataset.targets = [normal_class] * dataset.data.shape[0]
    print("Normal Train Data: ", dataset.data.shape)

    os.makedirs("./Dataset/CIFAR10/test", exist_ok=True)
    test_set = CIFAR10("./Dataset/CIFAR10/test", train=False, download=True, transform=img_transform)
    print("Test Train Data:", test_set.data.shape)

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=32,
        shuffle=False,
    )

    return train_dataloader, test_dataloader



# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# https://github.com/amazon-science/spot-diff/blob/main/utils/prepare_data.py
# def _mkdirs_if_not_exists(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# def VisA():
#     data_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2',
#                 'pcb3', 'pcb4', 'pipe_fryum']
#     split_type = '1cls'
#     split_file = 'datasets/VisA/split_csv/1cls.csv' #'./split_csv/1cls.csv'
#     data_folder = 'datasets/VisA/'
#     save_folder = os.path.join('./VisA_pytorch/', split_type)

#     if split_type == '1cls':
#         for data in data_list:
#             train_folder = os.path.join(save_folder, data, 'train')
#             test_folder = os.path.join(save_folder, data, 'test')
#             mask_folder = os.path.join(save_folder, data, 'ground_truth')

#             train_img_good_folder = os.path.join(train_folder, 'good')
#             test_img_good_folder = os.path.join(test_folder, 'good')
#             test_img_bad_folder = os.path.join(test_folder, 'bad')
#             test_mask_bad_folder = os.path.join(mask_folder, 'bad')

#             _mkdirs_if_not_exists(train_img_good_folder)
#             _mkdirs_if_not_exists(test_img_good_folder)
#             _mkdirs_if_not_exists(test_img_bad_folder)
#             _mkdirs_if_not_exists(test_mask_bad_folder)

#         with open(split_file, 'r') as file:
#             csvreader = csv.reader(file)
#             header = next(csvreader)
#             for row in csvreader:
#                 object, set, label, image_path, mask_path = row
#                 if label == 'normal':
#                     label = 'good'
#                 else:
#                     label = 'bad'
#                 image_name = image_path.split('/')[-1]
#                 mask_name = mask_path.split('/')[-1]
#                 img_src_path = os.path.join(data_folder, image_path)
#                 msk_src_path = os.path.join(data_folder, mask_path)
#                 img_dst_path = os.path.join(save_folder, object, set, label, image_name)
#                 msk_dst_path = os.path.join(save_folder, object, 'ground_truth', label, mask_name)
#                 shutil.copyfile(img_src_path, img_dst_path)
#                 if set == 'test' and label == 'bad':
#                     mask = Image.open(msk_src_path)

#                     # binarize mask
#                     mask_array = np.array(mask)
#                     mask_array[mask_array != 0] = 255
#                     mask = Image.fromarray(mask_array)

#                     mask.save(msk_dst_path)
#     else:
#         for data in data_list:
#             train_folder = os.path.join(save_folder, data, 'train')
#             test_folder = os.path.join(save_folder, data, 'test')
#             mask_folder = os.path.join(save_folder, data, 'ground_truth')
#             train_mask_folder = os.path.join(mask_folder, 'train')
#             test_mask_folder = os.path.join(mask_folder, 'test')

#             train_img_good_folder = os.path.join(train_folder, 'good')
#             train_img_bad_folder = os.path.join(train_folder, 'bad')
#             test_img_good_folder = os.path.join(test_folder, 'good')
#             test_img_bad_folder = os.path.join(test_folder, 'bad')

#             train_mask_bad_folder = os.path.join(train_mask_folder, 'bad')
#             test_mask_bad_folder = os.path.join(test_mask_folder, 'bad')

#             _mkdirs_if_not_exists(train_img_good_folder)
#             _mkdirs_if_not_exists(train_img_bad_folder)
#             _mkdirs_if_not_exists(test_img_good_folder)
#             _mkdirs_if_not_exists(test_img_bad_folder)
#             _mkdirs_if_not_exists(train_mask_bad_folder)
#             _mkdirs_if_not_exists(test_mask_bad_folder)

#         with open(split_file, 'r') as file:
#             csvreader = csv.reader(file)
#             header = next(csvreader)
#             for row in csvreader:
#                 object, set, label, image_path, mask_path = row
#                 if label == 'normal':
#                     label = 'good'
#                 else:
#                     label = 'bad'
#                 image_name = image_path.split('/')[-1]
#                 mask_name = mask_path.split('/')[-1]
#                 img_src_path = os.path.join(data_folder, image_path)
#                 msk_src_path = os.path.join(data_folder, mask_path)
#                 img_dst_path = os.path.join(save_folder, object, set, label, image_name)
#                 msk_dst_path = os.path.join(save_folder, object, 'ground_truth', set, label, mask_name)
#                 shutil.copyfile(img_src_path, img_dst_path)
#                 if label == 'bad':
#                     mask = Image.open(msk_src_path)

#                     # binarize mask
#                     mask_array = np.array(mask)
#                     mask_array[mask_array != 0] = 255
#                     mask = Image.fromarray(mask_array)

#                     mask.save(msk_dst_path)