import torch
import torch.nn as nn

# 모델 정의하는부분
class UNetDEC(nn.Module):
    def __init__(self,output_channel):
        super(UNetDEC , self).__init__()
        def CBR2d(in_channels , out_channels , kernel_size = 3 , stride = 1 , padding = 1 , bias = True):
            '''
            Conv + Batch Norm + ReLU
            same w,h
            '''
            layers = []
            layers += [nn.Conv2d(in_channels = in_channels ,out_channels = out_channels , 
                                kernel_size = kernel_size , stride = stride , padding = padding , 
                                bias = bias)]
            layers += [nn.BatchNorm2d(num_features = out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)
            return cbr

        def make_layer_dec(in_channels , out_channels ):
            '''
            '''
            layer_1  = CBR2d(in_channels , in_channels // 2)
            layer_2 = CBR2d(in_channels // 2 , in_channels // 2)
            unpool = nn.ConvTranspose2d(in_channels=in_channels // 2, out_channels=out_channels,kernel_size=2, stride=2, padding=0, bias=True)
            return nn.Sequential(layer_1 , layer_2 ,unpool)


        self.dec5 = nn.ConvTranspose2d(in_channels=2048 , out_channels=1024,kernel_size=2, stride=2, padding=0, bias=True)
        self.dec4 = make_layer_dec(2* 1024 , 512)
        self.dec3 = make_layer_dec(2* 512 , 256)
        self.dec2 = make_layer_dec(2* 256 , 128)
        self.fc0 = nn.ConvTranspose2d(in_channels=128 , out_channels=64,kernel_size=2, stride=2, padding=0, bias=True)
        self.fc = nn.Conv2d(in_channels=64, out_channels=output_channel, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self , levels):
        x = levels[-1]
        out = self.dec5(x)
        out = torch.cat((out, levels[-2]) , dim = 1) # in (2048,32,32) , out (512,64,64)
        out = self.dec4(out) # 512 , 64, 64
        out = torch.cat((out , levels[-3]) , dim = 1) 
        out = self.dec3(out) # in (1024,64,64) , out (256,128,128)
        out = torch.cat((out , levels[-4]) , dim = 1) 
        out = self.dec2(out) # in (256,128,128) , out (128,256,256)
        out = self.fc0(out) #  in (128,256,256) , out (64, 512,512)
        out = self.fc(out)

        return out
    

if __name__=="__main__":
    backbone_outputs = [torch.randn(1,256,128,128) ,
                        torch.randn(1,512,64,64) ,
                        torch.randn(1,1024,32,32) ,
                        torch.randn(1,2048,16,16) ]
    dec = UNetDEC()
    torch.onnx.export(dec,  backbone_outputs, 'unetDEC.onnx', 
                    input_names = ['level1' , 'level2' , 'level_3' , 'level_4'] , 
                    output_names = ['preds'], opset_version = 11 
                  )