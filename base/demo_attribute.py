import torch

dev = torch.device("cpu")
# dev = torch.device("cuda")
a = torch.tensor([2, 2],
                 dtype=torch.float32,
                 device=dev)
print(a)

i = torch.tensor([[0, 1, 2], [0, 1, 2]])
v = torch.tensor([1, 2, 3])
a = torch.sparse_coo_tensor(i, v, (4, 4),
                            dtype=torch.float32,
                            device=dev).to_dense()
print(a)

indices = torch.tensor([[4, 2, 1], [2, 0, 2]])
values = torch.tensor([3, 4, 5], dtype=torch.float32)
# x = torch.sparse_coo_tensor(indices=indices, values=values, size=[5, 5])
x = torch.sparse_coo_tensor(indices, values, (5, 5), device=dev).to_dense()
print(x)
