# pyremote

Pythonic remote code execution using context managers. Run code on remote servers as if it were local.

## How to

1. Install,

```bash
pip3 install git+https://github.com/Scicom-AI-Enterprise-Organization/pyremote
```

2. Simple execution,

```python
from pyremote import remote

x = 10
y = [1, 2, 3]

with remote("192.168.1.100", "ubuntu", password="secret"):
    result = x * 2
    y.append(result)
    print(f"Computed: {result}")

print(y) 
```

It will automatically sync local variables with remote variables.