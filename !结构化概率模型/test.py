import itertools
import random
import time
import psutil
import os

n = 25 # 变量的个数
k = 2  # 每个变量可以取的值数

# 记录开始时间和初始内存
start_time = time.time(
process = psutil.Process(os.getpid())
start_memory = process.memory_info().rss / (1024 * 1024)  # 转换为MB

# 初始化表格，使用字典存储联合分布
P = {}

# 生成所有可能的组合并赋予随机概率
for combo in itertools.product(range(k), repeat=n):
    P[combo] = random.random()  # 为每个组合赋予随机概率

# 记录结束时间和最终内存
end_time = time.time()
end_memory = process.memory_info().rss / (1024 * 1024)  # 转换为MB

# 计算时间和内存开销
execution_time = end_time - start_time
memory_usage = end_memory - start_memory

# 输出时间和内存开销
print(f"建立表格的时间开销: {execution_time:.6f} 秒")
print(f"建立表格的内存开销: {memory_usage:.6f} MB")

# 示例：输出部分联合分布
print("\n部分联合分布:")
for i, (key, value) in enumerate(P.items()):
    print(f"P{key} = {value:.6f}")
    if i >= 5:  # 仅输出前 6 项
        break