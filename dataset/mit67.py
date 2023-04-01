import os
import scipy.io
import csv
import numpy as np
from os.path import join
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, list_dir


def find_classes(dir):
    fname = os.path.join(dir, 'TrainImages.txt')
    # read the content of the file
    with open(fname) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    # find the list of classes
    classes = dict()
    for x in content:
        classes[x.split("/")[0]] = 0

    # assign a label for each class
    index = 0
    for key in sorted(classes):
        classes[key] = index
        index += 1

    return classes

def make_dataset(dir, classes, train):
    images = []

    if train:
        fname = os.path.join(dir, 'TrainImages.txt')
    else:
        fname = os.path.join(dir, 'TestImages.txt')

    # read the content of the file
    with open(fname) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    for x in content:
        path = x
        label = classes[x.split("/")[0]]
        item = (path, label)
        images.append(item)

    return images


def write_csv_file(dir, images, train):
    if train:
        file_name = 'train'
    else:
        file_name = 'test'
    csv_file = os.path.join(dir, file_name + '.csv')
    if not os.path.exists(csv_file):

        # write a csv file
        print('[dataset] write file %s' % csv_file)
        with open(csv_file, 'w') as csvfile:
            fieldnames = ['name', 'label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for x in images:
                writer.writerow({'name': x[0], 'label': x[1]})

        csvfile.close()


class Mit67(VisionDataset):
    """`Stanford mit67 <http://web.mit.edu/torralba/www/indoor/>`_ Dataset.
        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    urls = {
        'images': 'http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar',
        'train_file': 'http://web.mit.edu/torralba/www/TrainImages.txt',
        'test_file': 'http://web.mit.edu/torralba/www/TestImages.txt'
    }

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Mit67, self).__init__(root, transform=transform, target_transform=target_transform)

        self.root = root
        self.loader = default_loader
        self.train = train

        self.transform = transform
        self.target_transform = target_transform
        self.path_images = os.path.join(self.root, 'Images')


        self.classes = find_classes(self.root)
        self.images = make_dataset(self.root, self.classes, self.train)


        #write_csv_file(self.root, self.images, self.train)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name, target = self.images[index]
        image_path = join(self.path_images, image_name)
        image = self.loader(image_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

class Mit67Sample(VisionDataset):
    """`Stanford mit67 <http://web.mit.edu/torralba/www/indoor/>`_ Dataset.
        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    urls = {
        'images': 'http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar',
        'train_file': 'http://web.mit.edu/torralba/www/TrainImages.txt',
        'test_file': 'http://web.mit.edu/torralba/www/TestImages.txt'
    }

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, k=4096):
        super(Mit67Sample, self).__init__(root, transform=transform, target_transform=target_transform)

        self.root = root
        self.loader = default_loader
        self.train = train
        self.k = k

        self.transform = transform
        self.target_transform = target_transform
        self.path_images = os.path.join(self.root, 'Images')


        self.classes = find_classes(self.root)
        self.images = make_dataset(self.root, self.classes, self.train)


        #write_csv_file(self.root, self.images, self.train)
        num_classes = 67
        num_samples = len(self.images)
        label = np.zeros(num_samples, dtype=np.int32)
        for i in range(num_samples):
            _, target = self.images[i]
            label[i] = target
        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name, target = self.images[index]
        image_path = join(self.path_images, image_name)
        image = self.loader(image_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # sample contrastive examples
        pos_idx = index
        neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
        sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
        # sample_idx = np.hstack((neg_idx))
        return image, target, index, sample_idx




if __name__ == '__main__':
    train_dataset = Mit67('./mit67', train=True, download=False)
    test_dataset = Mit67('./mit67', train=False, download=False)