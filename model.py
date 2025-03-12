import torch
import torch.nn as nn

# kernel_size, layers, padding, stride
architecture_config = [
    (3, 32, 1, 1),
    (3, 64, 1, 1),
    (3, 128, 1, 1)
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyReLu = nn.LeakyReLU(0.1)  # Fix alpha value

    def forward(self, x):
        return self.leakyReLu(self.batchnorm(self.conv(x)))

class ModelAI(nn.Module):
    def __init__(self, in_channels=1, **kwargs):
        super(ModelAI, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.backbone = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fullyconnection(**kwargs)

    def forward(self, x):
        x = self.backbone(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if isinstance(x, tuple):
                layers.append(
                    CNNBlock(in_channels=in_channels,
                             out_channels=x[1],
                             kernel_size=x[0],
                             padding=x[2],
                             stride=x[3])
                )
                in_channels = x[1]
            else:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        return nn.Sequential(*layers)

    def _create_fullyconnection(self, dataSize, layers):
        return nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.0),
            nn.Sigmoid(),
            nn.Linear(dataSize * dataSize * layers, 1600),
            nn.LeakyReLU(0.1)
        )

# Kiểm tra mô hình trên CUDA
# def test():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     model = ModelAI(in_channels=1, dataSize=40, layers=128).to(device)
#     x = torch.randn(2, 1, 40, 40).to(device)  # Đưa dữ liệu lên GPU
#     output = model(x)

#     print(output.shape)
# test()

model = ModelAI(in_channels=1, dataSize=40, layers=128)
total_params = sum(p.numel() for p in model.parameters())
print(f"Tổng số tham số của mô hình: {total_params}")