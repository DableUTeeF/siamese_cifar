from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader
import numpy as np


class SiameseCifarLoader:
    def __init__(self, root):
        self.trainset = self.DataSet(CIFAR10(root=root, train=True, download=True),
                                     )
        self.testset = self.DataSet(CIFAR10(root=root, train=False, download=True),
                                    )

    def get_trainset(self, batch_size, num_worker, shuffle=True):
        return self.Loader(self.trainset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_worker)

    def get_testset(self, batch_size, num_worker):
        return self.Loader(self.testset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=num_worker)

    class DataSet:
        def __init__(self, dset):
            self.dset = dset
            self.set_length = len(dset)
            self.curidx = -1

        def __next__(self):
            self.curidx += 1
            return self[self.curidx]

        def __len__(self):
            return self.set_length

        def __getitem__(self, idx):
            x = np.zeros((2, 32, 32, 3), dtype='uint8')

            # first image
            x_1, y_1 = self.dset[idx]
            x[0] = np.array(x_1, dtype='uint8')
            # second image
            x_2, y_2 = self.dset[np.random.randint(0, self.set_length - 1)]
            x[1] = np.array(x_2, dtype='uint8')
            y = y_1 != y_2

            # preprocess
            x = x.astype('float32')
            x /= 255.
            x = np.rollaxis(x, 3, 1)
            return x, y

    class Loader(DataLoader):
        def __len__(self):
            return int(np.round(len(self.dataset) / self.batch_size))
