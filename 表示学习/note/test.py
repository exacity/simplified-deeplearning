import numpy as np
import matplotlib.pyplot as plt

# 基向量
h1 = np.array([1, 0, 0])
h2 = np.array([0, 1, 0])
h3 = np.array([0, 0, 1])

# 输入向量
x = np.array([0.5, 0.5, 0])

# 计算在基向量方向上的投影
projection_h1 = np.dot(x, h1)
projection_h2 = np.dot(x, h2)

# 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制基向量
ax.quiver(0, 0, 0, h1[0], h1[1], h1[2], color='r', label='h1')
ax.quiver(0, 0, 0, h2[0], h2[1], h2[2], color='g', label='h2')
ax.quiver(0, 0, 0, h3[0], h3[1], h3[2], color='b', label='h3')

# 绘制输入向量
ax.quiver(0, 0, 0, x[0], x[1], x[2], color='k', label='x')

ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])
ax.legend()
plt.show()
