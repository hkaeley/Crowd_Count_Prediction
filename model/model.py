import torch.nn as nn
import torch

class SimpleCrowdModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.layer1 = nn.Sequential(
            # 1. define a COnv2d layer with parameters: input_channel, output_channel, kernel_size, stride, padding
            # input_channel: the input
            # ouput channel:  how many feature map we are generating
            # kernel size: Convolutional filter size
            # padding: zero padding      
            nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding), ##
            
            # 2. define a BatchNorm2d layer for normalizing the output from the previous layer input, to let the layer nn learn independently, improving performance
            nn.BatchNorm2d(num_features = out_channels), ##)

            # 3. define a ReLU activation
            ##
            nn.ReLU(),

            # 4. define a MaxPooling layer for dividing current feature map size by the maxpool kernel size
            nn.MaxPool2d(kernel_size = 3, stride = 1) #stride = None by default ##)
            )

        self.layer2 = nn.Sequential(
            # Layer2 should have the same network structure of layer1
            # 1. define a COnv2d layer with parameters: input_channel, output_channel, kernel_size, stride, padding
             nn.Conv2d(in_channels = out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding), ##
            
            # 2. define a BatchNorm2d layer for normalizing the output from the previous layer input, to let the layer nn learn independently, improving performance
            nn.BatchNorm2d(num_features = out_channels), ##)

            # 3. define a ReLU activation
            ##
            nn.ReLU(),

            # 4. define a MaxPooling layer for dividing current feature map size by the maxpool kernel size
            nn.MaxPool2d(kernel_size = 3, stride = 1) #stride = None by default ##)
            )

        self.layer3 = nn.Sequential(
            # Layer2 should have the same network structure of layer1
            # 1. define a COnv2d layer with parameters: input_channel, output_channel, kernel_size, stride, padding
            nn.Conv2d(in_channels = out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding), ##
            
            # 2. define a BatchNorm2d layer for normalizing the output from the previous layer input, to let the layer nn learn independently, improving performance
            nn.BatchNorm2d(num_features = out_channels), ##)

            # 3. define a ReLU activation
            nn.ReLU(),

            # 4. define a MaxPooling layer for dividing current feature map size by the maxpool kernel size
            nn.MaxPool2d(kernel_size = 3, stride = 1) #stride = None by default ##)
            )
           
        
        self.layer4 = nn.Sequential(
            # Layer2 should have the same network structure of layer1
            # 1. define a COnv2d layer with parameters: input_channel, output_channel, kernel_size, stride, padding
             nn.Conv2d(in_channels = out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding), ##
            
            # 2. define a BatchNorm2d layer for normalizing the output from the previous layer input, to let the layer nn learn independently, improving performance
            nn.BatchNorm2d(num_features = out_channels), ##)

            # 3. define a ReLU activation
            ##
            nn.ReLU(),

            # 4. define a MaxPooling layer for dividing current feature map size by the maxpool kernel size
            nn.MaxPool2d(kernel_size = 3, stride = 1) #stride = None by default ##)
            )
        
        
        # define output layer which flattens the convolutional layer output
        # this layer layer indicates how many pixel positions we have in our output feature map from the latest convolution layer

        #each conv layer pretty much takes away 4 from height and width due to the conv2d and maxpool2d layers (see their formulas for more info)
        # therefore last lin layer needs to take as input out_channels * (initial_height - num_layers * 4) * (initial_width - num_layers * 4) 
        initial_height, initial_width = 28, 28
        num_layers = 4
        self.input_fts = out_channels * (initial_height - num_layers * 4) * (initial_width - num_layers * 4) 
        self.fc = nn.Linear(in_features = self.input_fts, out_features=1) #one out features since its regression


    def forward(self, x):
      out = self.layer1(x)
      out = self.layer2(out)
      out = self.layer3(out)
      out = self.layer4(out)
      
      #must resize before passing into linear layer
    #   out = out.reshape(batch_size, self.input_fts) #TODO: implement batch training on desktop later
      out = self.fc(out)
      return out
