from torch import nn

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
class VGG_19(nn.Module):
    def __init__(self, outputlayer=[26], num_classes=1000):
        assert (type(outputlayer)==list)
        super(VGG_19, self).__init__()
        layers = []
        in_channels = 3
        self.outputlayer = outputlayer
        for v in cfg:
            if v == 'M':

                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )


    def forward(self, x):
        for module in self.features._modules.values():
            x = module(x)

            if self.features._modules.values().index(module) in self.outputlayer:
                break
        return x