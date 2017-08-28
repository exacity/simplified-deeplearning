import numpy as np


def generate_data(w, b, instance_num):
  X = np.random.uniform(-100, 100, (w.shape[0], instance_num))
  y = np.matmul(X.T, w) + b

  return X, y

if __name__ == '__main__':
  generate_data(np.arange(10).reshape(-1, 1),1,199)