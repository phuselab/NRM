import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, Subset
import os
import numpy as np
import pandas as pd
import random
from collections import Counter
from PIL import Image

# THE IMAGENET FUNCTIONS USE HARDCODED FILE PATHS. THESE NEED TO CHANGE ESPECTIALLY FOR TESTING 

def get_test_dataloaders_stl10(args):
    """
    This code will add white noise(gaussian) to STL-10 images with 50% probability
    
    """
    train_loader, val_loader, test_loader = None, None, None

    if args.mode == 'noise':
        train_set = datasets.STL10(args.data_root, split='train',download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        AddGaussianNoise(0.,args.noise)
                                    ]))
        val_set = datasets.STL10(args.data_root, split='test',download=True,
                                transform=transforms.Compose([
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor(),
                                    AddGaussianNoise(0.,args.noise)
                                ]))

    if args.mode == 'blur':
        train_set = datasets.STL10(args.data_root, split='train',download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        AddGaussianBlur(7, args.blur)
                                    ]))
        val_set = datasets.STL10(args.data_root, split='test',download=True,
                                transform=transforms.Compose([
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor(),
                                    AddGaussianBlur(7, args.blur)
                                ]))

    if args.mode == 'color':
        if args.color == 'gray':
            train_set = datasets.STL10(args.data_root, split='train',download=True,
                                        transform=transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Grayscale(num_output_channels=3),
                                            transforms.ToTensor(),
                                        ]))
            val_set = datasets.STL10(args.data_root, split='test',download=True,
                                    transform=transforms.Compose([
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                    ]))
        elif args.color == 'color':
            train_set = datasets.STL10(args.data_root, split='train',download=True,
                                        transform=transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                        ]))
            val_set = datasets.STL10(args.data_root, split='test',download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]))

    if args.use_valid:
        train_set_index = torch.randperm(len(train_set))
        if os.path.exists(os.path.join(args.save, 'index.pth')):
            print('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load(os.path.join(args.save, 'index.pth'))
        else:
            print('!!!!!! Save train_set_index !!!!!!')
            torch.save(train_set_index, os.path.join(args.save, 'index.pth'))
        if args.data.startswith('cifar'):
            num_sample_valid = 5000
        elif args.data == 'stl10':
            num_sample_valid = 4000
        else:
            num_sample_valid = 50000

        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[:-num_sample_valid]),
                num_workers=args.workers, pin_memory=True)
        if 'val' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[-num_sample_valid:]),
                num_workers=args.workers, pin_memory=True)
        if 'test' in args.splits:
            test_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
    else:
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        if 'val' or 'test' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            test_loader = val_loader

    return train_loader, val_loader, test_loader


#JE ADDED
def get_test_dataloaders_imagenet(args):
    """
    This code will add white noise(gaussian) to ImageNet images with 50% probability
    
    """
    train_loader, val_loader, test_loader = None, None, None

    if args.mode == 'noise':
        val_set = datasets.ImageNet('/misc/vlgscratch4/LecunGroup/katrina/data/imagenet_split/', 
                                split='val',download=False,
                                transform=transforms.Compose([
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    AddGaussianNoise(0.,args.noise)
                                ]))

    if args.mode == 'blur':
        val_set = datasets.ImageNet('/misc/vlgscratch4/LecunGroup/katrina/data/imagenet_split/', 
                                split='val',download=False,
                                transform=transforms.Compose([
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    AddGaussianBlur(49, args.blur)
                                ]))

    if args.mode == 'color':
        if args.color == 'gray':
            val_set = datasets.ImageNet('/misc/vlgscratch4/LecunGroup/katrina/data/imagenet_split/', 
                                    split='val',download=False,
                                    transform=transforms.Compose([
                                        transforms.Resize((224,224)),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                    ]))
        elif args.color == 'color':
            val_set = datasets.ImageNet('/misc/vlgscratch4/LecunGroup/katrina/data/imagenet_split/', 
                                    split='val',download=False,
                                    transform=transforms.Compose([
                                        transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                    ]))

    test_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return test_loader

def get_human_test_dataloaders_imagenet(args):
	class HumanImagesDataset(Dataset):
		def __init__(self, root_dir, dirs = ['0', '1', '2', '3', '4'], transform=None):
			self.root_dir = root_dir
			self.transform = transform
			self.dirs = dirs
			self.classes = ['knife', 'keyboard', 'elephant', 'bicycle', 'airplane', 'clock', 
			'oven', 'chair', 'bear', 'boat', 'cat', 'bottle', 'truck', 'car', 'bird', 'dog']
			self.names = dict([(t, os.listdir(os.path.join(root_dir, t))) for t in os.listdir(root_dir) if t in self.dirs])
			
		def __len__(self):
			return len([f for t in self.names for f in self.names[t]])

		def get_t_from_i(self, idx):
			for t in self.dirs:
				if idx < len(self.names[t]):
					return t
				else:
					idx -= len(self.names[t])

			return None

		def __getitem__(self, idx):
			t = self.get_t_from_i(idx)
			
			image_path = os.path.join(self.root_dir, t, self.names[t][idx])
            
			if 'color_gray' in image_path:
				mode = self.names[t][idx].split('_')[3] 
			else:
				mode = self.names[t][idx].split('_')[2]
			image = np.array(Image.open(image_path).convert('RGB'))
            # image = io.imread(image_path)

			label = self.classes.index(self.names[t][idx].split('_')[-1].split('.')[0])

			if self.transform:
				image = self.transform(image)

			return image, label, mode

	if args.mode == 'color':
		root_dir = '../datasets/ColorGraySplit'
	elif args.mode == 'noise':
		root_dir = '../datasets/NoiseSplit_gray_contrast0.2'
	elif args.mode == 'blur':
		root_dir = '../datasets/BlurSplit_color'

	dataset = HumanImagesDataset(root_dir, transform=transforms.ToTensor(), dirs = args.dirs)

	if args.subsample == False:
		dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
	else:
		print("TAKING SUBSET")
		idxs = random.sample(range(0, len(dataset)), len(dataset)//2)
		subset = Subset(dataset, idxs)
		print(len(subset))
		dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

	return dataloader


def get_test_dataloaders_cifar10(args):
    """
    This code will add white noise(gaussian) to CIFAR-10 images with 50% probability
    
    """
    train_loader, val_loader, test_loader = None, None, None

    if args.mode == 'noise':
        train_set = datasets.CIFAR10(args.data_root, train=True,download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        AddGaussianNoise(mean=0., std=1., contrast=0.1)
                                    ]))
        val_set = datasets.CIFAR10(args.data_root, train=False,download=True,
                                transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        AddGaussianNoise(mean=0., std=1., contrast=0.1)
                                    ]))

    if args.mode == 'blur':
        train_set = datasets.CIFAR10(args.data_root, train=True,download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        AddGaussianBlur(7, args.blur)
                                    ]))
        val_set = datasets.CIFAR10(args.data_root, train=False,download=True,
                                transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        AddGaussianBlur(7, args.blur)
                                ]))

    if args.mode == 'color':
        if args.color == 'gray':
            train_set = datasets.CIFAR10(args.data_root, train=True,download=True,
                                        transform=transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Grayscale(num_output_channels=3),
                                            transforms.ToTensor(),
                                        ]))
            val_set = datasets.CIFAR10(args.data_root, train=False,download=True,
                                    transform=transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Grayscale(num_output_channels=3),
                                            transforms.ToTensor(),
                                        ]))
        elif args.color == 'color':
            train_set = datasets.CIFAR10(args.data_root, train=True,download=True,
                                        transform=transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                        ]))
            val_set = datasets.CIFAR10(args.data_root, train=False,download=True,
                                    transform=transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                        ]))

    if args.use_valid:
        train_set_index = torch.randperm(len(train_set))
        if os.path.exists(os.path.join(args.save, 'index.pth')):
            print('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load(os.path.join(args.save, 'index.pth'))
        else:
            print('!!!!!! Save train_set_index !!!!!!')
            torch.save(train_set_index, os.path.join(args.save, 'index.pth'))
        if args.data.startswith('cifar'):
            num_sample_valid = 5000
        else:
            num_sample_valid = 50000

        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[:-num_sample_valid]),
                num_workers=args.workers, pin_memory=True)
        if 'val' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[-num_sample_valid:]),
                num_workers=args.workers, pin_memory=True)
        if 'test' in args.splits:
            test_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
    else:
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        if 'val' or 'test' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            test_loader = val_loader

    return train_loader, val_loader, test_loader


def get_train_dataloaders_stl10(args):
    """
    This code will add white noise(gaussian) to STL-10 images with 50% probability
    
    """
    train_loader, val_loader, test_loader = None, None, None
    # normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
    #                                  std=[0.2471, 0.2435, 0.2616])
    if args.mode == 'noise':
        train_set = datasets.STL10(args.data_root, split='train',download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(96, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        AllRandomNoise()
                                    ]))
        val_set = datasets.STL10(args.data_root, split='test',download=True,
                                transform=transforms.Compose([
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor(),
                                    AllRandomNoise()
                                ]))

    if args.mode == 'blur':
        train_set = datasets.STL10(args.data_root, split='train',download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(96, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        AllRandomBlur(7)
                                    ]))
        val_set = datasets.STL10(args.data_root, split='test',download=True,
                                transform=transforms.Compose([
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor(),
                                    AllRandomBlur(7)
                                ]))

    if args.mode == 'color':
        if args.color == 'gray':
            train_set = datasets.STL10(args.data_root, split='train',download=True,
                                        transform=transforms.Compose([
                                            transforms.RandomCrop(96, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Grayscale(num_output_channels=3),
                                            transforms.ToTensor(),
                                        ]))
            val_set = datasets.STL10(args.data_root, split='test',download=True,
                                    transform=transforms.Compose([
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                    ]))
        elif args.color == 'color':
            train_set = datasets.STL10(args.data_root, split='train',download=True,
                                        transform=transforms.Compose([
                                            transforms.RandomCrop(96, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                        ]))
            val_set = datasets.STL10(args.data_root, split='test',download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]))

    if args.use_valid:
        train_set_index = torch.randperm(len(train_set))
        if os.path.exists(os.path.join(args.save, 'index.pth')):
            print('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load(os.path.join(args.save, 'index.pth'))
        else:
            print('!!!!!! Save train_set_index !!!!!!')
            torch.save(train_set_index, os.path.join(args.save, 'index.pth'))
        if args.data.startswith('cifar'):
            num_sample_valid = 5000
        elif args.data == 'stl10':
            num_sample_valid = 4000
        else:
            num_sample_valid = 50000

        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[:-num_sample_valid]),
                num_workers=args.workers, pin_memory=True)
        if 'val' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[-num_sample_valid:]),
                num_workers=args.workers, pin_memory=True)
        if 'test' in args.splits:
            test_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
    else:
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        if 'val' or 'test' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            test_loader = val_loader

    return train_loader, val_loader, test_loader

#JE ADDED
def get_train_dataloaders_imagenet(args):
    """
    This code will add white noise(gaussian) to Imagenet images with 50% probability
    
    """
    train_loader, val_loader, test_loader = None, None, None
    # normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
    #                                  std=[0.2471, 0.2435, 0.2616])
    if args.mode == 'noise':
        train_set = datasets.ImageNet('/misc/vlgscratch4/LecunGroup/katrina/data/imagenet_split/', 
                                    split='train',download=False,
                                    transform=transforms.Compose([
                                        transforms.Resize((224,224)),
                                        transforms.RandomCrop(224, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        AllRandomNoise(0., contrast=0.2)
                                    ]))
        val_set = datasets.ImageNet('/misc/vlgscratch4/LecunGroup/katrina/data/imagenet_split/', 
                                split='val',download=False,
                                transform=transforms.Compose([
                                    transforms.Resize((224,224)),
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor(),
                                    AllRandomNoise(0., contrast=0.2)
                                ]))

    if args.mode == 'blur':
        train_set = datasets.ImageNet('/misc/vlgscratch4/LecunGroup/katrina/data/imagenet_split/', 
                                    split='train',download=False,
                                    transform=transforms.Compose([
                                        transforms.Resize((224,224)),
                                        transforms.RandomCrop(224, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        # transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        AllRandomBlur(kernel=49, std=1.0)
                                    ]))
        val_set = datasets.ImageNet('/misc/vlgscratch4/LecunGroup/katrina/data/imagenet_split/', 
                                split='val',download=False,
                                transform=transforms.Compose([
                                    transforms.Resize((224,224)),
                                    # transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor(),
                                    AllRandomBlur(kernel=49, std=1.0)
                                ]))

    if args.mode == 'color':
        if args.color == 'gray':
            train_set = datasets.ImageNet('/misc/vlgscratch4/LecunGroup/katrina/data/imagenet_split/', 
                                        split='train',download=False,
                                        transform=transforms.Compose([
                                            transforms.Resize((224,224)),
                                            transforms.RandomCrop(224, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Grayscale(num_output_channels=3),
                                            transforms.ToTensor(),
                                        ]))
            val_set = datasets.ImageNet('/misc/vlgscratch4/LecunGroup/katrina/data/imagenet_split/',
                                    split='val',download=False,
                                    transform=transforms.Compose([
                                        transforms.Resize((224,224)),
                                            transforms.RandomCrop(224, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Grayscale(num_output_channels=3),
                                            transforms.ToTensor(),
                                    ]))
        elif args.color == 'color':
            train_set = datasets.ImageNet('/misc/vlgscratch4/LecunGroup/katrina/data/imagenet_split/', 
                                        split='train',download=False,
                                        transform=transforms.Compose([
                                            transforms.Resize((224,224)),
                                            transforms.RandomCrop(224, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                        ]))
            val_set = datasets.ImageNet('/misc/vlgscratch4/LecunGroup/katrina/data/imagenet_split/', 
                                    split='val',download=False,
                                    transform=transforms.Compose([
                                        transforms.Resize((224,224)),
                                        transforms.RandomCrop(224, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                    ]))

    # Compute frequencies of each of 16 categories and therefore, determine sampling weights
    train_targets = train_set.targets
    target_freqs1000 = dict(Counter(train_targets))
    sixteen_class_map_master = pd.read_csv('../datasets/imagenet_mapping.csv', header=0)
    sixteen_class_mapping = pd.Series(sixteen_class_map_master['sixteen_class_id'].values,
                                        index=sixteen_class_map_master['thousand_class_id']).to_dict()
    train_targets16 = [sixteen_class_mapping[i] for i in train_targets if i in sixteen_class_mapping]
    target_freqs16 = dict(Counter(train_targets16))
    weights1000 = []
    for i in list(set(train_targets)):
        if i in sixteen_class_mapping:
            w = 1. / (target_freqs16[sixteen_class_mapping[i]] * target_freqs1000[i])
        else:
            w = 0.
        weights1000.append(w)

    weights = [weights1000[i] for i in train_targets]
    train_sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_set), replacement=True)

    if args.use_valid:
        train_set_index = torch.randperm(len(train_set))
        if os.path.exists(os.path.join(args.save, 'index.pth')):
            print('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load(os.path.join(args.save, 'index.pth'))
        else:
            print('!!!!!! Save train_set_index !!!!!!')
            torch.save(train_set_index, os.path.join(args.save, 'index.pth'))
        if args.data.startswith('cifar'):
            num_sample_valid = 5000
        elif args.data == 'stl10':
            num_sample_valid = 4000
        else:
            num_sample_valid = 50000 #assuming this is correct for imagenet?

        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[:-num_sample_valid]),
                num_workers=args.workers, pin_memory=True)
        if 'val' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[-num_sample_valid:]),
                num_workers=args.workers, pin_memory=True)
        if 'test' in args.splits:
            test_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
    else:
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, sampler=train_sampler,
                num_workers=args.workers, pin_memory=True)
        if 'val' or 'test' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            test_loader = val_loader

    return train_loader, val_loader, test_loader


def get_train_dataloaders_cifar10(args):
    """
    This code will add white noise(gaussian) to CIFAR-10 images with 50% probability
    
    """
    train_loader, val_loader, test_loader = None, None, None
    # normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
    #                                  std=[0.2471, 0.2435, 0.2616])
    if args.mode == 'noise':
        train_set = datasets.CIFAR10(args.data_root, train=True,download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        AddGaussianNoise(mean=0., std=1., contrast=0.1)
                                    ]))
        val_set = datasets.CIFAR10(args.data_root, train=False,download=True,
                                transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        AddGaussianNoise(mean=0., std=1., contrast=0.1)
                                ]))

    if args.mode == 'blur':
        train_set = datasets.CIFAR10(args.data_root, train=True,download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        AllRandomBlur(7)
                                    ]))
        val_set = datasets.CIFAR10(args.data_root, train=False,download=True,
                                transform=transforms.Compose([
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor(),
                                    AllRandomBlur(7)
                                ]))

    if args.mode == 'color':
        if args.color == 'gray':
            train_set = datasets.CIFAR10(args.data_root, train=True,download=True,
                                        transform=transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Grayscale(num_output_channels=3),
                                            transforms.ToTensor(),
                                        ]))
            val_set = datasets.CIFAR10(args.data_root, train=False,download=True,
                                    transform=transforms.Compose([
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                    ]))
        elif args.color == 'color':
            train_set = datasets.CIFAR10(args.data_root, train=True,download=True,
                                        transform=transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                        ]))
            val_set = datasets.CIFAR10(args.data_root, train=False,download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]))

    if args.use_valid:
        train_set_index = torch.randperm(len(train_set))
        if os.path.exists(os.path.join(args.save, 'index.pth')):
            print('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load(os.path.join(args.save, 'index.pth'))
        else:
            print('!!!!!! Save train_set_index !!!!!!')
            torch.save(train_set_index, os.path.join(args.save, 'index.pth'))
        if args.data.startswith('cifar'):
            num_sample_valid = 5000
        else:
            num_sample_valid = 50000

        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[:-num_sample_valid]),
                num_workers=args.workers, pin_memory=True)
        if 'val' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[-num_sample_valid:]),
                num_workers=args.workers, pin_memory=True)
        if 'test' in args.splits:
            test_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
    else:
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        if 'val' or 'test' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            test_loader = val_loader

    return train_loader, val_loader, test_loader

class AllRandomBlur(object):
    def __init__(self, kernel=7, std=1.0):
        self.kernel = kernel
        self.std = std
    
    def __call__(self, tensor):
        if self.std != 0.0:
            tensor = transforms.GaussianBlur(kernel_size = self.kernel,sigma=(0.00001,self.std))(tensor)

        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddGaussianBlur(object):
    def __init__(self, kernel=7, std=1.0):
        self.kernel = kernel
        self.std = std
    
    def __call__(self, tensor):
        if self.std != 0.0:
            tensor = transforms.GaussianBlur(kernel_size = self.kernel, sigma=self.std)(tensor)

        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddGaussianNoise(object):
    """
    Author: Omkar Kumbhar
    Description:
    Adding gaussian noise to images in the batch
    """
    def __init__(self, mean=0., std=1., contrast=0.1):
        self.std = std
        self.mean = mean
        self.contrast = contrast

    def __call__(self, tensor):
        noise = torch.Tensor()
        n = tensor.size(1) * tensor.size(2)
        sd2 = self.std * 2

        while len(noise) < n:
            # more samples than we require
            m = 2 * (n - len(noise))
            new = torch.randn(m) * self.std

            # remove out-of-range samples
            new = new[new >= -sd2]
            new = new[new <= sd2]

            # append to noise tensor
            noise = torch.cat([noise, new])
        
        # pick first n samples and reshape to 2D
        noise = torch.reshape(noise[:n], (tensor.size(1), tensor.size(2)))

        # stack noise and translate by mean to produce std + 
        newnoise = torch.stack([noise, noise, noise]) + self.mean

        # shift image hist to mean = 0.5
        tensor = tensor + (0.5 - tensor.mean())

        # self.contrast = 1.0 / (5. * max(1.0, tensor.max() + sd2, 1.0 + (0 - tensor.min() - sd2)))
        # print(self.contrast)

        tensor = transforms.functional.adjust_contrast(tensor, self.contrast)
        
        return tensor + newnoise + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AllRandomNoise(object):
    def __init__(self, mean=0., std=0.04, contrast=0.1):
        self.std = std
        self.mean = mean
        self.contrast = contrast
        self.all_devs = np.arange(0.0,std + 0.01,0.01)
    
    def __call__(self, tensor):
        self.std = np.random.choice(self.all_devs)
        noise = torch.Tensor()
        n = tensor.size(1) * tensor.size(2)
        sd2 = self.std * 2

        while len(noise) < n:
            # more samples than we require
            m = 2 * (n - len(noise))
            new = torch.randn(m) * self.std

            # remove out-of-range samples
            new = new[new >= -sd2]
            new = new[new <= sd2]

            # append to noise tensor
            noise = torch.cat([noise, new])
        
        # pick first n samples and reshape to 2D
        noise = torch.reshape(noise[:n], (tensor.size(1), tensor.size(2)))

        # stack noise and translate by mean to produce std + 
        newnoise = torch.stack([noise, noise, noise]) + self.mean

        # shift image hist to mean = 0.5
        tensor = tensor + (0.5 - tensor.mean())

        # self.contrast = 1.0 / (5. * max(1.0, tensor.max() + sd2, 1.0 + (0 - tensor.min() - sd2)))
        # print(self.contrast)

        tensor = transforms.functional.adjust_contrast(tensor, self.contrast)
        
        return tensor + newnoise + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

    
class RandomGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        """
        Apply Gaussian noise if a number between 1 and 10 is less or equal than 5
        """
        if random.randint(1,10) <= 5:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Clip(object):
    def __init__(self, min_=0., max_=1.):
        self.min_ = min_
        self.max_ = max_
        
    def __call__(self, tensor):
        """
        Clamp values in given range
        """
        return torch.clamp(tensor, min=self.min_, max=self.max_)
    