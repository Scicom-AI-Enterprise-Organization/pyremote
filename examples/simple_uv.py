from pyremote import remote, UvConfig
import numpy as np

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

    df = pd.DataFrame({'name': ['a', 'b', 'c'], 'data': np.array([1, 2, 3])})

    return df

result = compute()
print(result)