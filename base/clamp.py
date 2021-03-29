import torch

a = torch.rand(2, 2) * 10

print(a)
#
#       | min, if x_i < min
# y_i = | x_i, if min <= x_i <= max
#       | max, if x_i > max
a = a.clamp(2, 5)

print(a)
