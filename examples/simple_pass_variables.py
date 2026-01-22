from pyremote import remote, UvConfig

@remote(
    "localhost", 
    "ubuntu", 
    password="ubuntu123", 
    uv=UvConfig(path="~/.venv", python_version="3.10.17"), 
    dependencies=["numpy==1.26.4", "pandas==2.2.3"]
)
def compute(a, b=10):
    import numpy as np
    result = np.array([1, 2, 3]) * a + b
    print(f"a={a}, b={b}, result={result}")
    return result.tolist()

result1 = compute(5)
print(result1)

result2 = compute(2, b=100)
print(result2)

result3 = compute(a=3, b=50)
print(result3)