class BaseBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(BaseBlock, self).__init__()
        self.subsampling = 2 if in_channels != out_channels else 1
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.subsampling, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):

        out = self.sequential(x)
        return out

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, base_block = BaseBlock):

        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.base_block = base_block(in_channels, out_channels)
        self.block_expantion = self.base_block.subsampling
        self.shortcut = nn.Identity()
        if self.base_block.subsampling==2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride=self.block_expantion,
                          padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )



    def forward(self, x):

        residual = self.shortcut(x)
        out = self.base_block(x)
        out += residual
        out = F.relu(out)

        return out
