from pyremote import remote, UvConfig
import numpy as np
import sys

@remote(
    "localhost", 
    "ubuntu", 
    password="ubuntu123", 
    uv=UvConfig(path="~/.venv", python_version="3.10.17", install_uv=False), 
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
    uv=UvConfig(path="~/.venv-3.12", python_version="3.12", install_uv=False), 
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