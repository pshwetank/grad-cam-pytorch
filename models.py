import torchvision.models as models
import torch
import torch.nn
import torch.nn.functional as F

#Using two parallel CNNs :
#Inspiration: DeepInsight
class deepinsight_mobilenetv2(torch.nn.Module):
    def __init__(self):
        super(deepinsight_mobilenetv2, self).__init__()
        self.net1 = models.mobilenet_v2(pretrained = True)
        self.net2 = models.mobilenet_v2(pretrained = True)
        self.fc_comb = torch.nn.Linear(in_features = 2000, out_features = 500)
        self.fc_last = torch.nn.Linear(in_features = 500, out_features = 2)
    def forward(self, x):
        x1 = self.net1(x)
        x2 = self.net2(x)
        #print(x1.shape)
        #print(x2.shape)
        x = torch.cat((x1,x2),dim=1)
        x = self.fc_comb(x) 
        x = F.relu(x)
        x = self.fc_last(x)
        x = F.relu(x)
        return x

class MobileNet_v2(torch.nn.Module):
    def __init__(self):
        super(MobileNet_v2,self).__init__()
        self.vanilla_model = models.mobilenet_v2(pretrained = True)
        self.vanilla_model.classifier[1] = torch.nn.Linear(in_features = self.vanilla_model.classifier[1].in_features, out_features = 1)
    def forward(self,x):
        x = self.vanilla_model(x)
        x = torch.sigmoid(x)
        return x
    
class MobileNet_v2_TL(torch.nn.Module):
    def __init__(self, n_fclayer=256, dropout_prob = 0.4):
        super(MobileNet_v2_TL,self).__init__()
        self.vanilla_model = models.mobilenet_v2(pretrained = True)
        for param in self.vanilla_model.parameters():
            param.requires_grad = False
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(1000, n_fclayer),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(n_fclayer, 1)
        )
    def forward(self,x):
        x = self.vanilla_model(x)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x

class Ghostnet_TL(torch.nn.Module):
    def __init__(self, n_fclayer=256, dropout_prob = 0.4):
        super(Ghostnet_TL, self).__init__()
        self.vanilla_ghostnet = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
        for param in self.vanilla_ghostnet.parameters():
            param.requires_grad = False
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(1000, n_fclayer),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(n_fclayer, 1)
        )
    def forward(self, x):
        x = self.vanilla_ghostnet(x)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x

class SqueezeNet_TL(torch.nn.Module):
    def __init__(self):
        super(SqueezeNet_TL, self).__init__()
        self.vanilla_snet = models.squeezenet1_0(pretrained = True)
        for param in self.vanilla_snet.parameters():
            param.requires_grad = False
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(1000, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, 1)
        )
    def forward(self, x):
        x = self.vanilla_snet(x)
        x = fc1(x)
        x = torch.sigmoid(x)
        return x

class Ghostnet_proto(torch.nn.Module):
    def __init__(self, n_fclayer=256, dropout_prob = 0.4):
        super(Ghostnet_proto, self).__init__()
        self.vanilla_ghostnet = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(1000, n_fclayer),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(n_fclayer, 1)
        )
    def forward(self, x):
        x = self.vanilla_ghostnet(x)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x