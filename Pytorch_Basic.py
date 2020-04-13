import torch

# empty matrix
x = torch.empty(5,3)

# rand matrix
x = torch.rand(5,3)
x = torch.randint(high=5,size = (5,3))

print(x)

# zero matrix
x=torch.zeros(5,3,dtype=torch.long)

x = torch.tensor([5.5,3])

x = torch.new_ones(5,3,dtype = torch.double)
x = torch.randn_like(x, dtype=torch.float)