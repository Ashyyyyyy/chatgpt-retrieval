# README

## Dependencies

```bash
pip install chromadb

# Langchain
pip install langchain langchain_community langchain_core langchain_text_splitters langchain_chroma

# DirectoryLoader
pip install "unstructued[xlsx]"
pip install "unstructured[docx]"
pip install tqdm

# thrift

pip install thrift
```

## Directory Tree

- `pdfschat_cn_queue.py` 主程序

- `constant.py` 存放API_KEYS

- `data/` 存储所有文档的原始文件，可以时pdf，txt，word，excel。

- `SEserver_queue.py` 服务器程序

初次运行后会生成的缓存文件：

- `embeddings_cache/` 存放所有embeddings

- `persist/` 存放vectorstore（在sql中）

- `file_name_dictionary/` 存放记录文件变化的logs

- `record_manager_cache.sql` 存放记录vectorstore变化的index，更新时可以仅更新变化的条目。

## Thrift

```bash
thrift -r --gen js:node searchEngine.thrift
thrift -r --gen py searchEngine.thrift
```
