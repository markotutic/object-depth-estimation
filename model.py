import os.path

import torch

import sdk.python.sintel_io


class ObjectDepthModule(torch.nn.Module):
    def __init__(self):
        self.coarse1 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=4)
        self.coarse1pool = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.coarse2 = torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5))
        self.coarse2pool = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.coarse3 = torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3,3))

        self.coarse4 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3))

        self.coarse5 = torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3))

        self.fc1 = torch.nn.Linear(256 * 1, 4096)
        self.fc2 = torch.nn.Linear(4096, 1)
        # output global depth image

        self.fine1 = torch.nn.Conv2d(in_channels=3, out_channels=63, kernel_size=(9,9), stride=2)
        self.fine1pool = torch.nn.MaxPool2d(kernel_size=(2,2))

        self.fine3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding=2, padding_mode="zeros")
        self.fine4 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(5, 5), padding=2, padding_mode="zeros")

    def forward(self, x):
        c1 = self.coarse1(x)
        c1 = self.coarse1pool(c1)
        c2 = self.coarse2(c1)
        c2 = self.coarse2pool(c2)
        c3 = self.coarse3(c2)
        c4 = self.coarse4(c3)
        c5 = self.coarse5(c4)
        c6 = self.fc1(c5.flatten())
        c6 = torch.relu(c6)
        c7 = self.fc2(c6)
        c7 = c7.reshape((5, 5, 1))

        f1 = self.fine1(x)
        f1 = self.fine1pool(x)
        f2 = torch.cat((c7, f1))
        f3 = self.fine3(f2)
        f4 = self.fine4(f3)


    def loss(self, x, x_gt):
        return

if __name__ == "__main__":
    depth = sdk.python.sintel_io.depth_read(os.path.abspath("training/depth/alley_1/frame_0001.dpt"))
    print(depth)


