# pyremote

Pythonic remote code execution using decorators. Run code on remote servers as if it were local. 

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

@remote("localhost", "ubuntu", password="ubuntu123", python_path="python3.10")
def compute():
    result = x * 2
    y.append(result)
    print(f"Computed: {result}")
    return result

r = compute()
print(y)  # [1, 2, 3, 20]
print(r)
```

Output,

```
Computed: 20
[1, 2, 3, 20]
20
```

## API

```python
def remote(
    host: str,
    username: str,
    port: int = 22,
    password: Optional[str] = None,
    key_filename: Optional[str] = None,
    key_password: Optional[str] = None,
    timeout: int = 30,
    venv: Optional[Union[str, VenvConfig]] = None,
    uv: Optional[Union[str, UvConfig]] = None,
    dependencies: List[str] = None,
    python_path: Optional[str] = None,
    setup_commands: List[str] = None,
) -> Callable:
    """
    Decorator for remote Python function execution over SSH.
    
    The decorated function will execute on the remote machine. Variables
    from the enclosing scope are automatically captured and modifications
    are synced back after execution.
    
    Args:
        host: Remote hostname or IP
        username: SSH username
        port: SSH port (default 22)
        password: SSH password (optional if using key)
        key_filename: Path to SSH private key (optional)
        key_password: Passphrase for SSH key (optional)
        timeout: Connection timeout in seconds
        venv: Standard venv path (str) or VenvConfig object
        uv: UV venv path (str) or UvConfig object (takes precedence over venv)
        dependencies: List of pip packages to install on remote
        python_path: Override python interpreter path
        setup_commands: List of shell commands to run before execution
    
    Note:
        To reassign variables from outer scope, use `global` keyword:
        
            x = 10
            
            @remote(...)
            def compute():
                global x  # required for reassignment
                x = x + 1
        
        Mutable objects (lists, dicts) can be modified in-place without `global`.
    
    Examples:
        # basic usage
        x = 10
        y = [1, 2, 3]
        
        @remote("server.com", "user", password="pass")
        def compute():
            y.append(x * 2)  # in-place modification works
            return x * 2
        
        result = compute()
        print(y)  # [1, 2, 3, 20]
        
        
        # with UV (auto-installs uv and creates venv)
        @remote("server.com", "user", password="pass",
                uv=UvConfig(path="~/.venv", python_version="3.12"),
                dependencies=["numpy", "pandas"])
        def process():
            import numpy as np
            import pandas as pd
            return np.array([1, 2, 3])
        
        
        # simple UV usage (uses defaults)
        @remote("server.com", "user", password="pass",
                uv="~/.venv",
                dependencies=["numpy"])
        def compute():
            import numpy as np
            return np.mean([1, 2, 3])
        
        
        # UV with specific Python version
        @remote("server.com", "user", password="pass",
                uv=UvConfig(path=".venv", python_version="3.11"),
                dependencies=["torch"])
        def train():
            import torch
            return torch.cuda.is_available()
        
        
        # standard venv
        @remote("server.com", "user", password="pass",
                venv="~/.venv",
                dependencies=["requests"])
        def fetch():
            import requests
            return requests.get("https://api.example.com").json()
    """
```

## More examples

### Pass variables

```python
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
```

Output,

```
a=5, b=10, result=[15 20 25]
[15, 20, 25]
a=2, b=100, result=[102 104 106]
[102, 104, 106]
a=3, b=50, result=[53 56 59]
[53, 56, 59]
```

Checkout [examples/simple_pass_variables.py](examples/simple_pass_variables.py)

### Global numpy array

```python
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
```

Output,

```
[2 3 4 2 3 4]
[3 4 5 3 4 5 3 4 5 3 4 5]
```

Checkout [examples/simple_numpy.py](examples/simple_numpy.py)

### Use UV

```python
from pyremote import remote, UvConfig
import numpy as np
import sys

@remote(
    "localhost", 
    "ubuntu", 
    password="ubuntu123", 
    uv=UvConfig(path="~/.venv", python_version="3.10.17"), 
    dependencies=["numpy==1.26.4", "pandas==2.2.3"]
)
def compute():
    import numpy as np
    import pandas as pd

    print('inside compute()', sys.version)
    df = pd.DataFrame({'name': ['a', 'b', 'c'], 'data': np.array([1, 2, 3])})

    return df

@remote(
    "localhost", 
    "ubuntu", 
    password="ubuntu123", 
    uv=UvConfig(path="~/.venv-3.12", python_version="3.12"), 
    dependencies=["numpy==1.26.4", "pandas==2.2.3"]
)
def compute2():
    import numpy as np
    import pandas as pd

    print('inside compute2()', sys.version)
    df = pd.DataFrame({'name': ['a', 'b', 'c'], 'data': np.array([1, 2, 3])})
    
    return df

result = compute()
print(result)
result = compute2()
print(result)
```

Output,

```
inside compute() 3.10.17 (main, Apr  9 2025, 08:54:15) [GCC 9.4.0]
  name  data
0    a     1
1    b     2
2    c     3
inside compute2() 3.12.12 (main, Dec  9 2025, 19:02:36) [Clang 21.1.4 ]
  name  data
0    a     1
1    b     2
2    c     3
```

Checkout [examples/simple_uv.py](examples/simple_uv.py)