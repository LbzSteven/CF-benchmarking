import time
from utils.cf_util import timeout

@timeout(5)
def long_running_function():
    import torch
    device = torch.device('cuda:1')
    a = torch.randn(10000, 10000, device=device)
    time.sleep(10)  # 模拟一个长时间运行的过程
    return a.sum().item()

start = time.time()
result = long_running_function()
print(time.time()-start)
print(result)