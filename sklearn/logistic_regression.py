from sklearn import linear_model

# reg = linear_model.LinearRegression()
# # X y
# reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
# print(reg.coef_)

# 岭系数最小化的是带罚项的残差平方和
reg = linear_model.Ridge(alpha=0.01)
reg.fit([[0, 0], [0, 0], [1, 1]], [0, 0.1, 1])
print(reg.coef_)
print(reg.intercept_)
