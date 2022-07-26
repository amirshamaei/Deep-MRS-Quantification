import torch


class ConvNet(torch.nn.Module):
    def __init__(self, depth, batch, outsize,kw):
        super().__init__()
        layers = []
        init = torch.nn.Sequential(
            torch.nn.Conv1d(2, 16, kw, stride=2), torch.nn.GELU(),
            torch.nn.Conv1d(16, 24, kw-2, stride=2), torch.nn.GELU())
        layers.append(init)
        inc = 24
        for i in range(0, depth):
            outc = inc + 8
            if batch == False:
                temp = torch.nn.Sequential(torch.nn.Conv1d(inc, outc, kw-2, stride=2), torch.nn.GELU())
            else:
                temp = torch.nn.Sequential(torch.nn.Conv1d(inc, outc, kw-2, stride=2), torch.nn.BatchNorm1d(outc),
                                           torch.nn.GELU())
            layers.append(temp)
            inc = outc
        layers.append(torch.nn.Flatten())
        self.encoder = torch.nn.Sequential(*layers)
        self.reg = torch.nn.Sequential(
                torch.nn.LazyLinear(outsize))

    def forward(self,x):
        x = self.encoder(x)
        x = self.reg(x)
        return x