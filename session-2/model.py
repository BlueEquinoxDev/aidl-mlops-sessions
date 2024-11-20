import torch.nn as nn

class MyModel(nn.Module):

    def __init__(self, dropout: float, kernel: int = 3, neurons: int = 512, act_func: str = "relu"):
        super().__init__()

        self.conv1 = nn.Conv2d(
                    in_channels=1,
                    out_channels=64,
                    kernel_size=(kernel, kernel),
                    padding=1,  
                )
        if act_func == "relu":
            self.act_func1 = nn.ReLU(inplace=True)
        elif act_func == "leakyrelu":
            self.act_func1 = nn.LeakyReLU(inplace=True)
        else:
            self.act_func1 = nn.Tanh()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=(kernel, kernel),
                    padding=1,    
                )
        if act_func == "relu":
            self.act_func2 = nn.ReLU(inplace=True)
        elif act_func == "leakyrelu":
            self.act_func2 = nn.LeakyReLU(inplace=True)
        else:
            self.act_func2 = nn.Tanh()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv3 = nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=(kernel, kernel),
                    padding=1,  
                )
        if act_func == "relu":
            self.act_func3 = nn.ReLU(inplace=True)
        elif act_func == "leakyrelu":
            self.act_func3 = nn.LeakyReLU(inplace=True)
        else:
            self.act_func3 = nn.Tanh()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = nn.Conv2d(                 
                    in_channels=256,
                    out_channels=256, 
                    kernel_size=(kernel, kernel),
                    padding=1,
                )
        if act_func == "relu":
            self.act_func4 = nn.ReLU(inplace=True)
        elif act_func == "leakyrelu":
            self.act_func4 = nn.LeakyReLU(inplace=True)
        else:
            self.act_func4 = nn.Tanh()
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        if kernel == 3:
            linear_layer = 4*4*256
        elif kernel == 4:
            linear_layer = 3*3*256
        else:
            linear_layer = 2*2*256

        self.mlp = nn.Sequential(
            
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(linear_layer, neurons), # 2*2*256, 512
            nn.ReLU(),
            nn.Dropout(dropout),  # <-- HERE
            nn.Linear(neurons, 15), # 512, 15
            nn.LogSoftmax(dim=-1)
        )


    def forward(self, x):
        #print(f"0: {x.shape}")
        x = self.maxpool1(self.act_func1(self.conv1(x)))
        #print(f"1: {x.shape}")

        x = self.maxpool2(self.act_func2(self.conv2(x)))
        #print(f"2: {x.shape}")

        x = self.maxpool3(self.act_func3(self.conv3(x)))
        #print(f"3: {x.shape}")

        x = self.maxpool4(self.act_func4(self.conv4(x)))
        #print(f"4: {x.shape}")

        x = self.mlp(x)
        #print(f"final: {x.shape}")

        return x
