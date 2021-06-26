import os
from torch.utils.data import Dataset
from PIL import Image
from scipy.io import loadmat
import numpy as np

class BSD500Dataset(Dataset):
    IMAGE_PATH, GT_PATH = 'IMAGE_PATH', 'GT_PATH'
    SEG_LABEL_TYPE = 'SEG_LABEL_TYPE'
    BOUNDARY_LABEL_TYPE = 'BOUNDARY_LABEL_TYPE'
    LABEL_TYPES = [SEG_LABEL_TYPE, BOUNDARY_LABEL_TYPE]

    FINE_MODE, COARSE_MODE, ALL_MODE = 'FINE_MODE', 'COARSE_MODE', 'ALL_MODE'
    MODES = [FINE_MODE, COARSE_MODE, ALL_MODE]
    SPLITS = ['train', 'test', 'val']

    def __init__(self, root, split='test', label=SEG_LABEL_TYPE, mode=COARSE_MODE):
        '''

        :param root: root directory of the dataset (parent dir of the actual data)
        :param split: train/val/test
        :param label: either boundary or segmentation
        :param mode: fine or coarse, or all of the available labels.
        '''
        assert split in self.SPLITS
        assert label in self.LABEL_TYPES
        assert mode in self.MODES

        self.split = split
        self.root = root
        self.label_type = label
        self.mode = mode

        self.images_dir = os.path.join(self.root, 'data', 'images', self.split)
        if not os.path.isdir(self.images_dir):  # in case you passed the base of the unzipped data as root
            self.root = f'{self.root}/BSR/BSDS500'
            self.images_dir = os.path.join(self.root, 'data', 'images', self.split)

        self.gt_dir = os.path.join(self.root, 'data', 'groundTruth', self.split)

        assert os.path.isdir(self.images_dir) and os.path.isdir(
            self.gt_dir), f'images or GTs images does not exist (imgs dir: {self.images_dir})'

        self.images = [os.path.join(self.images_dir, f) for f in os.listdir(self.images_dir)]
        self.gts = set([os.path.join(self.gt_dir, f) for f in os.listdir(self.gt_dir)])

        self.labeled_image_paths = []
        for im_path in sorted(self.images):
            im = os.path.basename(im_path).split('.')[0]
            gt_path = os.path.join(self.gt_dir, f'{im}.mat')
            if gt_path in self.gts:
                self.labeled_image_paths.append({self.IMAGE_PATH: im_path, self.GT_PATH: gt_path})

        self.num_samples = len(self.labeled_image_paths)

    def __getitem__(self, idx):
        paths = self.labeled_image_paths[idx]
        img_path = paths[self.IMAGE_PATH]
        gt_path = paths[self.GT_PATH]

        image = np.array(Image.open(img_path).convert('RGB'))
        raw_gt = loadmat(gt_path)
        gt = raw_gt['groundTruth'][0]

        gt_data = [gt[i][0, 0][self.LABEL_TYPES.index(self.label_type)] for i in range(gt.size)]

        item_data = {'im_path': img_path, 'gt_path': gt_path, 'im_name': os.path.basename(img_path), 'im': image}
        if self.mode == self.ALL_MODE:
            item_data['label'] = gt_data
        else:
            num_unique_values = [np.unique(g).size for g in gt_data]
            if self.mode == self.FINE_MODE:
                item_data['label'] = gt_data[np.argmax(num_unique_values)]  # map with the maximum number of segments
            elif self.mode == self.COARSE_MODE:
                item_data['label'] = gt_data[np.argmin(num_unique_values)]  # map with the least number of segments
        return item_data

    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    dataset_root = 'fill_your_own'
    ds = BSD500Dataset(dataset_root, split='test', mode='FINE_MODE')
    print(ds.__getitem__(0))
