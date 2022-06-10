import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.feature_dim == 0:
            self.net = timm.create_model(args.encoder_name, pretrained=args.pretrained,
                                         drop_path_rate=args.drop_path_rate, num_classes=args.num_classes)
        else: # For Self-supervised Learning
            self.net = timm.create_model(args.encoder_name, pretrained=args.pretrained,
                                         drop_path_rate=args.drop_path_rate, num_classes=args.feature_dim)
    def forward(self, x):
        return self.net(x)

# If you want to use this model in the another dataset, you have to change the model parameters.
class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.dfc3 = nn.Linear(z_dim, 4096)
        self.bn3 = nn.BatchNorm1d(4096)
        self.dfc2 = nn.Linear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.dfc1 = nn.Linear(4096, 256 * 3 * 3)
        self.bn1 = nn.BatchNorm1d(256 * 3 * 3)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.dconv5 = nn.ConvTranspose2d(256, 256, 3, padding=0)
        self.dconv4 = nn.ConvTranspose2d(256, 384, 3, padding=1)
        self.dconv3 = nn.ConvTranspose2d(384, 192, 3, padding=1)
        self.dconv2 = nn.ConvTranspose2d(192, 3, 5, padding=2)

    def forward(self, x):
        x = self.dfc3(x)
        x = F.relu(self.bn3(x))
        x = self.dfc2(x)
        x = F.relu(self.bn2(x))
        x = self.dfc1(x)
        x = F.relu(self.bn1(x))
        x = F.relu(x)
        x = x.view(-1, 256, 3, 3)
        x = self.upsample1(x)
        x = self.dconv5(x)
        x = F.relu(x)
        x = F.relu(self.dconv4(x))
        x = F.relu(self.dconv3(x))
        x = self.upsample1(x)
        x = self.dconv2(x)
        x = F.relu(x)
        x = self.upsample1(x)
        x = F.sigmoid(x)
        # print x
        return x

# For self-supervised learning
class Classification_Head(nn.Module):
    def __init__(self, args):
        super(Classification_Head, self).__init__()
        self.fc = nn.Linear(args.feature_dim, args.num_classes)

    def forward(self, x):
        return self.fc(x)

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, classifier):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier

    def forward(self, x, finetune=True):
        x = self.encoder(x)
        if finetune:
            return self.classifier(x)
        else:
            return self.decoder(x)