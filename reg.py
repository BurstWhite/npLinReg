import numpy as np
import matplotlib.pyplot as plt

# initialize
rng = np.random.default_rng(0)

n = int(input())
m = 100  # samples

real_w = rng.uniform(1, 10, (1, n))
real_b = rng.uniform(1, 10)

X = rng.uniform(1, 20, (n, m))
mu = X.mean(axis=1, keepdims=True)
sigma = X.std(axis=1, keepdims=True)
X = (X - mu) / sigma
# print(mu, sigma)
noise = rng.normal(0, 1, (1, m))
Y = real_w @ X + real_b + noise

w = np.zeros((1, n))
b = 0.0


# start gradient descent
lr = 0.001
losses = []
for i in range(500):
    l = w @ X + b - Y
    dw = (2/m) * l @ X.T
    db = (2/m) * np.sum(l)
    w -= lr * dw
    b -= lr * db
    losses.append((l**2).mean())

# theta, *_ = np.linalg.lstsq(np.vstack([X, np.ones((1, m))]).T, Y.T, rcond=None)
# w_ls = theta[:n].T
# b_ls = theta[n, 0]
# print(np.abs((real_w - w)).mean(), np.abs((w_ls - w)).mean())
# print(real_b - b, b_ls - b)


# generate plot
plt.plot(losses)
plt.xlabel("iteration")
plt.ylabel("MSE loss")
plt.yscale("log")
plt.title("Training loss")
plt.show()