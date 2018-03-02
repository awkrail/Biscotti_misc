from chainer.dataset import dataset_mixin
import glob
import numpy as np


#C DataSet
class ImgDCTDataset(dataset_mixin.DatasetMixin):
    def __init__(self, dataDir="./train/"):
        print("load dataset start...")
        print("from: %s" % dataDir)
        print("data_num: %d", len(glob.glob(dataDir)))
        self.dataDir = dataDir
        self.dataset = []

        train_npy_paths = glob.glob(self.dataDir)
        for train_npy_path in train_npy_path:
            data = np.load(train_npy_path)
            img, dct = data[:, :, :3], data[:, :, 3:]
            self.dataset.append((img, dct))
        print("load dataset done!")
    
    def __len__(self):
        return len(self.dataset)
    
    def get_example(self, i):
        return self.dataset[i][0], self.dataset[i][1] # YCbCr(input), Guetzli(DCT)