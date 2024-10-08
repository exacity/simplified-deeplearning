import networkx as nx
import matplotlib.pyplot as plt

# 创建一个有向图
G = nx.DiGraph()

# 添加节点
G.add_nodes_from([1, 2, 3, 4])

# 添加有向边
G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])

# 绘制图
nx.draw(G, with_labels=True, node_color='lightblue', arrowsize=20)
plt.show()
