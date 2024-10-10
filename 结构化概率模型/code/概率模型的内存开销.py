import itertools
import random
import time
import psutil
import os

n = 15 # 变量个数
k = 2 # 每个变量可以取的值的个数

start_time = time.time()
process = psutil.Process(os.getpid())
start_memory = process.memory_info().rss / (1024 * 1024)

P = {}
# 生成所有可能的组合并赋予随机概率
for combo in itertools.product(range(k), repeat=n):
    P[combo] = random.random()

end_time = time.time()
end_memory = process.memory_info().rss / (1024 * 1024)

execution_time = end_time - start_time
memory_usage = end_memory - start_memory

print(f"建立表格的时间开销: {execution_time:.6f} 秒")
print(f"建立表格的内存开销: {memory_usage:.6f} MB")

print("\n部分联合分布:") # 仅输出前5项
for i, (key, value) in enumerate(P.items()):
    print(f"P{key} = {value:.6f}")
    if i >= 4:
        break
