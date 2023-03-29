import torch
import torch.nn as nn
import torch.nn.init as init

from torchsummary import summary

def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

class AutoEncoder(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   #Conv1
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),  #Conv2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),  #Conv3
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  #Conv4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),  #Conv5
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), #Conv6
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, 3, stride=1, padding=1), #Conv7
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),  #Conv8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, d, 5, stride=1, padding=1), #Conv9
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d, 32, 5, stride=1, padding=1),    #Conv9_Inv
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 64, 3, stride=1, padding=1),     #Conv8_Inv
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 128, 3, stride=1, padding=1),    #Conv7_Inv
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),    #Conv6_Inv
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),     #Conv5_Inv
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),     #Conv4_Inv
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1),     #Conv3_Inv
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),     #Conv2_Inv
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),      #Conv1_Inv
        )

        init_weights(self.encoder.modules())
        init_weights(self.decoder.modules())

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)

        return y

if __name__ == "__main__":
    model = AutoEncoder().cuda()
    summary(model, (3, 128, 128))
