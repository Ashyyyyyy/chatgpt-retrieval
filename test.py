# 假设你的数据存储在一个set中
data = [("source1", 85), ("source2", 90), ("source3", 75), ("source1", 95), ("source3", 80)]
# data = set()  # 用于测试空数据集的情况
data = []

data.sort(key=lambda x: x[1], reverse=True)
