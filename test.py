from network.data import getDataLoader
from network.model import Encoder,Generator,Discriminator,Lenet
import torch
loader = getDataLoader()
encoder = Encoder()
model = Lenet()
for i,(input,label) in enumerate(loader):
    print(model(input).shape)
input = torch.randn([64,100])
label = torch.randn([64,2])
g = Generator()
d = Discriminator()
go = g(input,label)
do = d(go)
print(do)
