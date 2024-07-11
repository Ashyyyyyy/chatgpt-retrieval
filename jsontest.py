import json


def save_session_data(session_id, answer, source):
    # 数据字典
    data = {
        "session_id": session_id,
        "answer": answer,
        "source": source.split(',')  # 假设source是以逗号分隔的多个文件名
    }

    # JSON文件名
    file_name = f"{session_id}.json"

    # 写入JSON文件
    with open(file_name, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    print(f"Data saved to {file_name}")


# 示例数据
session_id = "123456"
answer = "test。"
source = "file1.txt,file2.pdf,file3.docx"

# 保存数据
save_session_data(session_id, answer, source)
