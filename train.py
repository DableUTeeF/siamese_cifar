from models import ResNet, ContrastiveLoss
from datagen import SiameseCifarLoader
import os
import json
from natthaphon import Model
import torch
from torch.nn import functional as F


class ThresholdAcc:
    def __call__(self, inputs, targets):
        distant = F.cosine_similarity(inputs[0], inputs[1])
        predict = (distant > 0.7).long()
        acc = torch.sum(predict != targets.long()).float() / targets.size(0)
        return acc

    def __str__(self):
        return 'acc()'


if __name__ == '__main__':
    model = Model(ResNet())
    model.compile(torch.optim.SGD(model.model.parameters(),
                                  lr=0.01,
                                  momentum=0.9,
                                  weight_decay=1e-4),
                  ContrastiveLoss(),
                  metric=ThresholdAcc(),
                  device='cuda')

    try:
        os.listdir('/root')
        rootpath = '/root/palm/DATA/'
    except PermissionError:
        rootpath = '/home/palm/PycharmProjects/DATA/'
    name = 'cifar10'
    datagen = SiameseCifarLoader(os.path.join(rootpath, name))
    train_generator = datagen.get_trainset(64, 1)
    val_geerator = datagen.get_testset(100, 1)
    h = model.fit_generator(train_generator, 200, validation_data=val_geerator, lrstep=[100, 150])
    model.save_weights('2.h5')
    with open('logs/2.json', 'w') as wr:
        json.dump(h, wr)
