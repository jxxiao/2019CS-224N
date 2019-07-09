import torch

x = torch.FloatTensor([[1,2],[3,4]], requires_grad=True)
v = torch.mean(x*x)

v.backward()
print(v.grad)
