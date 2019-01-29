from models import Model, ResNet, ContrastiveLoss
from datagen import SiameseCifarLoader
import os
import json


if __name__ == '__main__':
    model = Model(ResNet())
    model.compile('sgd', ContrastiveLoss())
    model.cuda()

    try:
        os.listdir('/root')
        rootpath = '/root/palm/DATA/'
    except PermissionError:
        rootpath = '/home/palm/PycharmProjects/DATA/'
    name = 'cifar10'
    datagen = SiameseCifarLoader(os.path.join(rootpath, name))
    train_generator = datagen.get_trainset(64, 8)
    val_geerator = datagen.get_testset(100, 8)
    h = model.fit_generator(train_generator, 200, validation_data=val_geerator, lrstep=[100, 150])
    model.save_weights('2.h5')
    with open('logs/2.json', 'w') as wr:
        json.dump(h, wr)
