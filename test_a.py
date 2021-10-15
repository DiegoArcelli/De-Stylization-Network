import torch
from ibn import ResNet_IBN, Bottleneck_IBN
from torchsummary import summary

# ds = DeStylizationModule()
# x = torch.rand(100, 3, 224, 224)
# y_1, y_2, y_3 = ds(x)
# print(y_1.shape, y_2.shape, y_3.shape)

model = ResNet_IBN(block=Bottleneck_IBN, layers=[3, 4, 6, 3], ibn_cfg=('a', 'a', 'a', None))
# for name, param in model.named_parameters():
#     print(name, param.size())
#     if name.split(".")[0] != "ds":
#         param.requires_grad = False
# summary(model)

# for param in list(model.parameters())[66:]:
#     # print(param.shape)
#     param.requires_grad = False
# summary(model)


for name, param in model.named_parameters():
    print(name, param.size())
    if name[:2] != "ds":
    # if name.split(".")[0] != "ds":
        param.requires_grad = False
    # else:
    #     print(name)
model.eval()
x = torch.randn(5, 3, 224, 224)
print(model)
y = model(x)
summary(model)
print(x.shape, y.shape)