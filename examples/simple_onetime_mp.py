from multiprocess import Pool
from pyremote import remote, UvConfig

def run_version(v):
    @remote(
        "localhost",
        "ubuntu",
        password="ubuntu123",
        uv=UvConfig(
            path=f"~/.venv-{v}",
            python_version=v,
            install_uv=True,
            delete_after_done=True
        ),
        dependencies=["numpy==1.26.4", "pandas==2.2.3"],
        install_verbose=True,
    )
    def compute():
        import numpy as np
        import pandas as pd
        import sys

        print("inside compute()", sys.version)
        return pd.DataFrame(
            {"name": ["a", "b", "c"], "data": np.array([1, 2, 3])}
        )

    return compute()

def main():
    versions = ["3.9", "3.10", "3.11", "3.12", "3.13"]

    with Pool(len(versions)) as pool:
        results = pool.map(run_version, versions)

    print(results)

if __name__ == "__main__":
    main()
