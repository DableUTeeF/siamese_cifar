import torch
from torch import nn
from torch.nn import functional as F
import time


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, last_relu, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.last_relu = last_relu

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.last_relu:
            x = F.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, False)
        self.res1 = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        y = self.conv1(x)
        y = F.max_pool2d(y, (3, 3), (2, 2), (1, 1))
        x = self.res1(x)
        y += x
        return y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        out += self.shortcut(x)
        out = out
        return out


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=(4, 4, 4)):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_once(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return out

    def forward(self, image_a, image_b):
        output1 = self.forward_once(image_a)
        output2 = self.forward_once(image_b)
        return output1, output2


class CosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_a, image_b):
        eps = 1e-10
        sum_support = torch.sum(torch.pow(image_a, 2), 1)
        support_manitude = sum_support.clamp(eps, float("inf")).rsqrt()
        dot_product = image_b.unsqueeze(1).bmm(image_a.unsqueeze(2)).squeeze()
        cosine_similarity = dot_product * support_manitude
        cosine_similarity = torch.sigmoid(cosine_similarity)
        return cosine_similarity


class Model:
    def __init__(self, model, optimizer=None, loss=None):
        self.model = model
        self.similarity = CosineSimilarity()
        self.optimizer = optimizer
        self.loss = loss
        self.device = 'cpu'

    def cuda(self):
        self.to('cuda')

    def cpu(self):
        self.to('cpu')

    def to(self, device):
        self.device = device
        self.model.to(self.device)
        self.similarity.to(self.device)
        self.loss.to(self.device)

    def compile(self, optimizer, loss):
        if optimizer in ['sgd', 'SGD']:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=0.1,
                                             momentum=0.9,
                                             weight_decay=1e-4)
        elif optimizer in ['adam', 'Adam']:
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=0.001,
                                              weight_decay=1e-4)
        else:
            assert isinstance(optimizer, torch.optim.optimizer.Optimizer), 'Optimizer should be an Optimizer object'
            self.optimizer = optimizer
        if loss is not None:
            self.loss = loss
        else:
            self.loss = nn.BCELoss()

    def fit_generator(self, generator, epoch, validation_data=None, lrstep=None):
        if self.loss is None:
            self.compile('sgd', None)
        if lrstep:
            schedule = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            lrstep)
        for e in range(epoch):
            self.lastext = ''
            self.start_epoch_time = time.time()
            self.last_print_time = self.start_epoch_time
            acc = 0
            total = 0
            total_loss = 0
            self.model.train()
            for idx, (inputs, targets) in enumerate(generator):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).float()
                inputs = inputs.permute(0, 1, 4, 2, 3).float()
                output_a, output_b = self.model(inputs[:, 0], inputs[:, 1])
                distant = self.similarity(output_a, output_b)
                loss = self.loss(distant, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                predict = torch.round(distant)
                acc += torch.sum(predict == targets).cpu().detach().numpy()
                total += targets.size(0)
                total_loss += loss.cpu().detach().numpy()
                self.fprint(acc / total, total_loss / (idx + 1), idx + 1, len(generator))
            self.fprint(acc / total, total_loss / len(generator))
            if validation_data:
                self.evaluate_generator(validation_data)
            if lrstep:
                schedule.step()

    def evaluate_generator(self, generator):
        if self.loss is None:
            self.compile('sgd', None)
        self.lastext = ''
        self.start_epoch_time = time.time()
        acc = 0
        total = 0
        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(generator):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).float()
                inputs = inputs.permute(0, 1, 4, 2, 3).float()
                output_a, output_b = self.model(inputs[:, 0], inputs[:, 1])
                distant = self.similarity(output_a, output_b)
                loss = self.loss(distant, targets)
                predict = torch.round(distant)
                acc += torch.sum(predict == targets).cpu().detach().numpy()
                total += predict.size(0)
                total_loss += loss.cpu().detach().numpy()
        self.fprint('', '', val_acc=acc / total, val_loss=total_loss / len(generator))

    def fprint(self, acc, loss, idx=0, len_genrator=0, val_acc=None, val_loss=None):
        if idx == 0 and len_genrator == 0:
            t1 = ''
            t2 = f'{time.time()-self.start_epoch_time:.4f} s/step - '
        else:
            t1 = f'{idx}/{len_genrator} - '
            t2 = f'{time.time() - self.last_print_time:.4f} s/step -'
        if val_acc is not None:
            t3 = f' - val_acc: {val_acc: .4f}'
            t4 = f' - val_loss: {val_loss: .4f}'
            print(t3 + t4)
            return
        else:
            t3 = t4 = ''
            print(' ' * len(self.lastext), end='\r')
        text = f'{t1+t2} acc: {acc: .4f} - loss: {loss: .4f}{t3+t4}'
        print(text, end='\r')
        self.last_print_time = time.time()
        self.lastext = text
