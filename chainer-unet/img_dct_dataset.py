from chainer.dataset import dataset_mixin
import glob
import numpy as np


#DataSet
class ImgDCTDataset(dataset_mixin.DatasetMixin):
    def __init__(self, dataDir="./train/", data_range=(1, 1000)):
        print("load dataset start...")
        print("from: %s" % dataDir + "*.npy")
        print("data_num: [%d, %d)" % (data_range[0], data_range[1]))
        self.dataDir = dataDir
        self.dataset = []
        
        paths = sorted(glob.glob(self.dataDir + "*.npy"))
        load_paths = paths[data_range[0]:data_range[1]]

        for load_path in load_paths:
            data = np.load(load_path)
            img, dct = data[:, :, :3], data[:, :, 3:]
            self.dataset.append((img, dct))
        import ipdb; ipdb.set_trace()
        print("load dataset done!")
    
    def __len__(self):
        return len(self.dataset)
    
    def get_example(self, i):
        return self.dataset[i][0], self.dataset[i][1] # YCbCr(input), Guetzli(DCT)