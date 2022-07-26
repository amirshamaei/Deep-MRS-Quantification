import torch


class MLPNet(torch.nn.Module):
    def __init__(self, depth, batch, outsize, kw):
        super().__init__()
        # self.patching = torch.nn.Conv1d(in_channels=2,out_channels=64,stride=3,kernel_size=kw)
        # self.patching = torch.nn.Sequential(torch.nn.LazyLinear(128),
        #                                     torch.nn.Linear(128, 32))
        layers = []
        init = torch.nn.Sequential(
            torch.nn.LazyLinear(768), torch.nn.GELU(),
            torch.nn.Linear(768,512), torch.nn.GELU())
        layers.append(init)
        inc = 512
        for i in range(0, depth):
            outc = int(inc/2)
            if batch == False:
                temp = torch.nn.Sequential(torch.nn.Linear(inc,outc), torch.nn.GELU())
            else:
                temp = torch.nn.Sequential(torch.nn.Linear(inc,outc), torch.nn.GELU(), torch.nn.BatchNorm1d(outc),
                                           torch.nn.GELU())
            layers.append(temp)
            inc = outc
        # layers.append(torch.nn.Flatten())
        # layers.append(torch.nn.Sequential(torch.nn.LazyLinear(128),
        #                                   torch.nn.Linear(128,128)))
        self.encoder = torch.nn.Sequential(*layers)
        self.reg = torch.nn.Sequential(torch.nn.Flatten(),
                torch.nn.LazyLinear(outsize))

    def forward(self,x):
        # x = x.permute(0, 2, 1)
        x = self.encoder(x)
        # x = x.permute(0, 2, 1)
        # x = self.patching(x)
        x = self.reg(x)
        return x