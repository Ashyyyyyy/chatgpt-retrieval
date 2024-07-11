import concurrent.futures
import time

a = [1, 2]

def invoke_chain():
    # 模拟一个需要花费时间的操作
    time.sleep(2)
    a.append(3)
    print("invoke task completed.")
    return "Task completed"

timeout = 1  # 超时时间为3秒

with concurrent.futures.ThreadPoolExecutor() as executor:
    future = executor.submit(invoke_chain)
    try:
        result = future.result(timeout=timeout)
        print(f"Result: {result}")
    except concurrent.futures.TimeoutError:
        print("The operation timed out")
