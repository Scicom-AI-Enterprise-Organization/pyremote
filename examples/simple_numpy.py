from pyremote import remote
import numpy as np

result = np.array([1, 2, 3])

@remote("localhost", "ubuntu", password="ubuntu123", python_path="python3.10", dependencies=["numpy==1.26.4"])
def compute():
    global result

    result = np.concatenate([result, result])
    result += 1

compute()
print(result)
compute()
print(result)