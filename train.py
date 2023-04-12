import model
import torch.nn as nn
import torch

backbone = model.resnet.ResNet50()
dec_head = model.unet.UNetDEC()

# pretrained
pretrained_weights = torch.load('./ckpts/pretrained/resnet50-19c8e357.pth')
pretrained_weights.pop('fc.weight')
pretrained_weights.pop('fc.bias')
backbone.load_state_dict(pretrained_weights)

class Seg_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = backbone
        self.dec_head = dec_head

    def forward(self, x):
        out = self.backbone(x)
        out = self.dec_head(out)
        return out


if __name__=="__main__":
    model = Seg_model()
    inputs = torch.randn(1,3,512,512)
    outputs = model(inputs)
    print(outputs.shape)