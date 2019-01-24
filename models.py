import torch
from torch import nn
from torch.nn import functional as F
import time
import sys
import numpy as np


class Progbar(object):
    """Displays a progress bar.

    # Arguments
        target: Total number of steps expected, None if unknown.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, interval=0.05):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.last_update = 0
        self.interval = interval
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose
        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)

    def update(self, current, values=None, force=False):
        """Updates the progress bar.

        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            force: Whether to force visual progress update.
        """
        values = values or []
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self.start)
        if self.verbose == 1:
            if (not force and (now - self.last_update) < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self.total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self.total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self.unique_values:
                info += ' - %s:' % k
                if isinstance(self.sum_values[k], list):
                    avg = np.mean(
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self.sum_values[k]

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += (' ' * (prev_total_width - self.total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self.unique_values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self.last_update = now

    def add(self, n, values=None):
        self.update(self.seen_so_far + n, values)


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
                                              lr=0.01,
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
        history = {'loss': [],
                   'acc': [],
                   }
        if validation_data:
            history['val_loss'] = []
            history['val_acc'] = []
        for e in range(epoch):
            print('Epoch:', e+1)
            self.lastext = ''
            self.start_epoch_time = time.time()
            self.last_print_time = self.start_epoch_time
            acc = 0
            total = 0
            total_loss = 0
            self.model.train()
            progbar = Progbar(len(generator))
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
                progbar.update(idx-1, [['loss', total_loss / (idx + 1)],
                                     ['acc', acc / total],
                                     ])
            metrix = [['loss', total_loss / len(generator)],
                      ['acc', acc / total],
                      ]
            history['acc'].append(acc / total)
            history['loss'].append(total_loss / len(generator))
            if validation_data:
                val_acc, val_total, val_loss, val_steps = self.evaluate_generator(validation_data)
                history['val_acc'].append(val_acc / val_total)
                history['val_loss'].append(val_loss / val_steps)
                metrix.append(['val_loss', (val_loss / val_steps)])
                metrix.append(['val_acc', (val_acc / val_total)])
            progbar.update(len(generator), metrix, force=True)
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
        return acc, total, total_loss, len(generator)
