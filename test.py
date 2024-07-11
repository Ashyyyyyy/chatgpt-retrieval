# 假设你的数据存储在一个set中
data = {("source1", 85), ("source2", 90), ("source3", 75), ("source1", 95), ("source3", 80)}
# data = set()  # 用于测试空数据集的情况

# 检查data是否为空
if not data:
    print("Data is empty.")
else:
    # 创建一个字典来存储source和最高的score
    source_dict = {}

    for source, score in data:
        if source not in source_dict or score > source_dict[source]:
            source_dict[source] = score

    # 将字典转换为列表
    merged_data = list(source_dict.items())

    # 根据score进行排序，升序排列
    sorted_data = sorted(merged_data, key=lambda x: x[1])

    # 如果需要降序排列，可以设置reverse=True
    # sorted_data = sorted(merged_data, key=lambda x: x[1], reverse=True)

    # 打印排序后的结果
    for source, score in sorted_data:
        print(f"Source: {source}, Score: {score}")
