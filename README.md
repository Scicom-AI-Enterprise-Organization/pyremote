# pyremote

Pythonic remote code execution using decorators. Run code on remote servers as if it were local, and support multiple Python versions at the same time!

## Features

1. Seamless Remote Execution, execute Python functions on remote servers using simple decorators.
2. Flexible Authentication, support for SSH password, key-based authentication, and key passphrases.
3. Multiple Python Versions, run different Python versions simultaneously using virtual environments or uv.
4. Automatic Dependency Management, auto-install pip packages on remote servers.
5. Variable State Synchronization, automatically sync global variables between local and remote execution.
6. Multi-GPU Support, built-in distributed training with PyTorch using multiprocessing.
7. Real-time Output Streaming, stream stdout/stderr with custom callbacks.
8. Environment Variable Control, set remote environment variables per function.
9. Also support for Jupyter Notebook, able to multi-remote.
10. Auto Cleanup, set to delete after done.

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
    host: str = None,
    username: str = None,
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
    install_verbose: bool = False,
    stdout_callback: Optional[Callable[[str], None]] = None,
    multiprocessing: Optional[Union[int, Literal["auto"], MultiprocessingConfig]] = None,
    env: Optional[Dict[str, str]] = None,
    jupyter_mode: bool = False,
    jupyter_profile: str = 'default',
    _return_executor: bool = False,
) -> Callable:
    """
    Decorator for remote Python function execution over SSH with optional
    multi-GPU/multi-process support. Can also be used to configure global
    executor for Jupyter notebook cell magic.

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
        install_verbose: If True, stream pip/uv install output
        stdout_callback: Callback for stdout streaming
        multiprocessing: Enable multi-GPU execution for PyTorch. Can be:
            - int: Explicit number of processes/GPUs
            - "auto": Auto-detect GPU count
            - MultiprocessingConfig: Full configuration
        env: Dictionary of environment variables to set on remote
        jupyter_mode: If True, automatically register %%remotecell magic and
            set up the remote environment immediately (connect, install dependencies).
            This makes the first cell execution faster.
        jupyter_profile: Profile name for Jupyter mode (default: 'default').
            Allows multiple remote connections. Use %%remotecell --profile <name>
            to execute on a specific profile.

    Examples:
        # As decorator - Single GPU execution (default)
        @remote("server", "user", password="xxx", dependencies=["torch"])
        def train():
            return "done"

        # With environment variables
        @remote("server", "user", password="xxx",
                dependencies=["torch"],
                env={
                    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                    "NCCL_DEBUG": "INFO",
                    "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
                    "HF_HOME": "/data/huggingface",
                })
        def train_with_env():
            import os
            print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
            return "done"

        # Multi-GPU with auto-detection
        @remote("server", "user", password="xxx",
                dependencies=["torch"],
                multiprocessing="auto")
        def train_ddp(config):
            # rank and world_size are automatically injected
            import torch
            # ... training code ...
            return {"loss": 0.1}  # only rank 0's return value is captured

        # Explicit GPU count
        @remote("server", "user", password="xxx",
                dependencies=["torch"],
                multiprocessing=4)
        def train_4gpu():
            ...

        # Full configuration
        @remote("server", "user", password="xxx",
                dependencies=["torch"],
                multiprocessing=MultiprocessingConfig(
                    num_processes=8,
                    backend="nccl",
                    master_port=29501,
                ))
        def train_custom(data):
            ...

        # With delete_after_done to clean up venv after execution
        @remote("server", "user", password="xxx",
                uv=UvConfig(path="~/.temp-venv", delete_after_done=True),
                dependencies=["numpy"])
        def one_time_job():
            import numpy as np
            return np.array([1, 2, 3])

        # In Jupyter notebooks - configure global executor (manual registration)
        from pyremote import remote, remotecell

        remote(
            host="server",
            username="user",
            password="xxx",
            dependencies=["torch", "numpy"]
        )
        remotecell()  # Register the magic

        # Or use jupyter_mode for automatic setup
        from pyremote import remote

        remote(
            host="server",
            username="user",
            password="xxx",
            uv=UvConfig(path="~/.venv-3.10", python_version="3.10"),
            dependencies=["torch", "numpy"],
            install_verbose=True,
            jupyter_mode=True  # Automatically register magic and setup environment
        )

        # Then use cell magic
        %%remotecell
        import torch
        print(f"PyTorch version: {torch.__version__}")
        result = torch.tensor([1, 2, 3])
        result  # This will be returned

        # Multiple profiles example
        remote("server1", "user", password="xxx", jupyter_mode=True, jupyter_profile="gpu1")
        remote("server2", "user", password="xxx", jupyter_mode=True, jupyter_profile="gpu2")

        %%remotecell --profile gpu1
        # Runs on server1

        %%remotecell --profile gpu2
        # Runs on server2

        %%remotecell
        # Runs on first registered profile (gpu1)
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

### Stdout callback

```python
from pyremote import remote, UvConfig

logs = []

def log_handler(line: str):
    """Callback that receives each line of stdout as it streams."""
    logs.append(line)
    # You could also send to logging, websocket, database, etc.
    # Example: logger.info(line)
    # Example: websocket.send(line)

@remote(
    "localhost", 
    "ubuntu", 
    password="ubuntu123",
    uv=UvConfig(path="~/.venv-3.12", python_version="3.12"),
    timeout=60,
    stdout_callback=log_handler,
)
def task_with_logging():
    import time
    import sys
    
    print(f"Python version: {sys.version}")
    print("Starting task...")
    
    for i in range(5):
        print(f"Processing step {i+1}/5")
        time.sleep(1)
    
    print("Task completed!")
    return {"status": "completed", "steps": 5}

if __name__ == "__main__":
    result = task_with_logging()
    
    print("\n--- Captured Logs ---")
    for log in logs:
        print(f"  > {log}")
    
    print(f"\nResult: {result}")
    print(f"Total log lines captured: {len(logs)}")

```

Output,

```
Python version: 3.12.12 (main, Dec  9 2025, 19:02:36) [Clang 21.1.4 ]
Starting task...
Processing step 1/5
Processing step 2/5
Processing step 3/5
Processing step 4/5
Processing step 5/5
Task completed!

--- Captured Logs ---
  > Python version: 3.12.12 (main, Dec  9 2025, 19:02:36) [Clang 21.1.4 ]
  > Starting task...
  > Processing step 1/5
  > Processing step 2/5
  > Processing step 3/5
  > Processing step 4/5
  > Processing step 5/5
  > Task completed!

Result: {'status': 'completed', 'steps': 5}
Total log lines captured: 8
```

Checkout [examples/simple_stdout_callback.py](examples/simple_stdout_callback.py)

### DDP Multi-GPUs

```python
from pyremote import remote, UvConfig

@remote(
    "localhost", 
    "ubuntu", 
    password="ubuntu123", 
    uv=UvConfig(path="~/.venv-3.12-v2", python_version="3.12", install_uv=True, delete_after_done=True), 
    dependencies=[
        "numpy==1.26.4", 
        "torch==2.9.1", 
        "transformers==4.57.3", 
        "accelerate",
        "datasets",
        "evaluate",
        "scikit-learn"
    ],
    install_verbose=True,
    multiprocessing=2,
    env={
        "NCCL_DEBUG": "INFO",
        "CUDA_VISIBLE_DEVICES": "0,1"
    },
)
def compute():
    import sys
    import os
    import torch
    import torch.distributed as dist
    import evaluate
    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        pipeline,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print('inside compute()', sys.version)
    print(f"[Rank {rank}/{world_size}] GPU: {torch.cuda.current_device()}")

    sms_dataset = load_dataset("sms_spam")
    sms_train_test = sms_dataset["train"].train_test_split(test_size=0.2)
    train_dataset = sms_train_test["train"]
    test_dataset = sms_train_test["test"]

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(
            examples["sms"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    seed = 22
    train_tokenized = train_dataset.map(tokenize_function)
    train_tokenized = train_tokenized.remove_columns(["sms"]).shuffle(seed=seed)
    test_tokenized = test_dataset.map(tokenize_function)
    test_tokenized = test_tokenized.remove_columns(["sms"]).shuffle(seed=seed)

    id2label = {0: "ham", 1: "spam"}
    label2id = {"ham": 0, "spam": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        label2id=label2id,
        id2label=id2label,
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    training_output_dir = "/tmp/sms_trainer"
    training_args = TrainingArguments(
        output_dir=training_output_dir,
        eval_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=8,
        max_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        compute_metrics=compute_metrics,
    )

    trainer.train()

compute()
```

Output,

```
inside compute() 3.12.12 (main, Dec  9 2025, 19:02:36) [Clang 21.1.4 ]
inside compute() 3.12.12 (main, Dec  9 2025, 19:02:36) [Clang 21.1.4 ]
[Rank 1/2] GPU: 1
[Rank 0/2] GPU: 0
Map: 100%|█████████████████████████| 4459/4459 [00:00<00:00, 7710.66 examples/s]
Map: 100%|█████████████████████████| 4459/4459 [00:00<00:00, 7719.84 examples/s]
Map: 100%|█████████████████████████| 1115/1115 [00:00<00:00, 7648.28 examples/s]
Map: 100%|█████████████████████████| 1115/1115 [00:00<00:00, 7671.47 examples/s]
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
husein-MS-7D31:1463326:1463326 [0] NCCL INFO Bootstrap: Using enp6s0:192.168.68.52<0>
husein-MS-7D31:1463326:1463326 [0] NCCL INFO cudaDriverVersion 12090
husein-MS-7D31:1463326:1463326 [0] NCCL INFO NCCL version 2.27.5+cuda12.9
husein-MS-7D31:1463326:1463326 [0] NCCL INFO Comm config Blocking set to 1
husein-MS-7D31:1463326:1463512 [0] NCCL INFO NET/Plugin: Could not find: libnccl-net.so. 
husein-MS-7D31:1463326:1463512 [0] NCCL INFO NET/IB : No device found.
husein-MS-7D31:1463326:1463512 [0] NCCL INFO NET/IB : Using [RO]; OOB enp6s0:192.168.68.52<0>
husein-MS-7D31:1463326:1463512 [0] NCCL INFO NET/Socket : Using [0]enp6s0:192.168.68.52<0> [1]tailscale0:100.97.85.61<0> [2]br-2b7f015e8963:172.20.0.1<0> [3]br-5c38f4f374ee:172.27.0.1<0> [4]br-67a6126956b9:172.19.0.1<0> [5]br-742293e338db:172.18.0.1<0> [6]ztrf2yld5r:172.28.203.92<0> [7]vethb485ae1:fe80::2034:8eff:fea5:3cc5%vethb485ae1<0> [8]veth4debc6c:fe80::942d:3fff:fe48:4386%veth4debc6c<0> [9]veth254fb75:fe80::544c:ccff:feb7:4816%veth254fb75<0> [10]vethadaf70f:fe80::2849:37ff:fe63:1cf9%vethadaf70f<0> [11]vethde259b1:fe80::f811:30ff:fe2d:7aa1%vethde259b1<0> [12]veth8505c3a:fe80::e055:a0ff:fe7a:4265%veth8505c3a<0> [13]veth49edb48:fe80::8c64:81ff:fe11:9dce%veth49edb48<0> [14]vethde8b77e:fe80::28d0:43ff:fe0b:593a%vethde8b77e<0> [15]veth1c0ef23:fe80::c00f:9eff:fe43:224e%veth1c0ef23<0>
husein-MS-7D31:1463326:1463512 [0] NCCL INFO Initialized NET plugin Socket
husein-MS-7D31:1463326:1463512 [0] NCCL INFO Assigned NET plugin Socket to comm
husein-MS-7D31:1463326:1463512 [0] NCCL INFO Using network Socket
husein-MS-7D31:1463326:1463512 [0] NCCL INFO ncclCommInitRankConfig comm 0x4452af40 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId 1000 commId 0x798dcd8b6f24bb47 - Init START
husein-MS-7D31:1463327:1463327 [1] NCCL INFO cudaDriverVersion 12090
husein-MS-7D31:1463327:1463327 [1] NCCL INFO Bootstrap: Using enp6s0:192.168.68.52<0>
husein-MS-7D31:1463327:1463327 [1] NCCL INFO NCCL version 2.27.5+cuda12.9
husein-MS-7D31:1463327:1463327 [1] NCCL INFO Comm config Blocking set to 1
husein-MS-7D31:1463327:1463515 [1] NCCL INFO NET/Plugin: Could not find: libnccl-net.so. 
husein-MS-7D31:1463327:1463515 [1] NCCL INFO NET/IB : No device found.
husein-MS-7D31:1463327:1463515 [1] NCCL INFO NET/IB : Using [RO]; OOB enp6s0:192.168.68.52<0>
husein-MS-7D31:1463327:1463515 [1] NCCL INFO NET/Socket : Using [0]enp6s0:192.168.68.52<0> [1]tailscale0:100.97.85.61<0> [2]br-2b7f015e8963:172.20.0.1<0> [3]br-5c38f4f374ee:172.27.0.1<0> [4]br-67a6126956b9:172.19.0.1<0> [5]br-742293e338db:172.18.0.1<0> [6]ztrf2yld5r:172.28.203.92<0> [7]vethb485ae1:fe80::2034:8eff:fea5:3cc5%vethb485ae1<0> [8]veth4debc6c:fe80::942d:3fff:fe48:4386%veth4debc6c<0> [9]veth254fb75:fe80::544c:ccff:feb7:4816%veth254fb75<0> [10]vethadaf70f:fe80::2849:37ff:fe63:1cf9%vethadaf70f<0> [11]vethde259b1:fe80::f811:30ff:fe2d:7aa1%vethde259b1<0> [12]veth8505c3a:fe80::e055:a0ff:fe7a:4265%veth8505c3a<0> [13]veth49edb48:fe80::8c64:81ff:fe11:9dce%veth49edb48<0> [14]vethde8b77e:fe80::28d0:43ff:fe0b:593a%vethde8b77e<0> [15]veth1c0ef23:fe80::c00f:9eff:fe43:224e%veth1c0ef23<0>
husein-MS-7D31:1463327:1463515 [1] NCCL INFO Initialized NET plugin Socket
husein-MS-7D31:1463327:1463515 [1] NCCL INFO Assigned NET plugin Socket to comm
husein-MS-7D31:1463327:1463515 [1] NCCL INFO Using network Socket
husein-MS-7D31:1463327:1463515 [1] NCCL INFO ncclCommInitRankConfig comm 0x35e5f030 rank 1 nranks 2 cudaDev 1 nvmlDev 1 busId 8000 commId 0x798dcd8b6f24bb47 - Init START
husein-MS-7D31:1463327:1463515 [1] NCCL INFO RAS client listening socket at 127.0.0.1<28028>
husein-MS-7D31:1463326:1463512 [0] NCCL INFO RAS client listening socket at 127.0.0.1<28028>
husein-MS-7D31:1463327:1463515 [1] NCCL INFO Bootstrap timings total 0.000375 (create 0.000014, send 0.000076, recv 0.000129, ring 0.000020, delay 0.000000)
husein-MS-7D31:1463326:1463512 [0] NCCL INFO Bootstrap timings total 0.792110 (create 0.000015, send 0.000072, recv 0.791850, ring 0.000009, delay 0.000000)
husein-MS-7D31:1463326:1463512 [0] NCCL INFO comm 0x4452af40 rank 0 nRanks 2 nNodes 1 localRanks 2 localRank 0 MNNVL 0
husein-MS-7D31:1463327:1463515 [1] NCCL INFO comm 0x35e5f030 rank 1 nRanks 2 nNodes 1 localRanks 2 localRank 1 MNNVL 0
husein-MS-7D31:1463326:1463512 [0] NCCL INFO Channel 00/02 : 0 1
husein-MS-7D31:1463327:1463515 [1] NCCL INFO Trees [0] -1/-1/-1->1->0 [1] -1/-1/-1->1->0
husein-MS-7D31:1463326:1463512 [0] NCCL INFO Channel 01/02 : 0 1
husein-MS-7D31:1463327:1463515 [1] NCCL INFO P2P Chunksize set to 131072
husein-MS-7D31:1463326:1463512 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1
husein-MS-7D31:1463326:1463512 [0] NCCL INFO P2P Chunksize set to 131072
husein-MS-7D31:1463327:1463515 [1] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so. 
husein-MS-7D31:1463326:1463512 [0] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so. 
husein-MS-7D31:1463326:1463512 [0] NCCL INFO Check P2P Type isAllDirectP2p 0 directMode 0
husein-MS-7D31:1463327:1463518 [1] NCCL INFO [Proxy Service] Device 1 CPU core 0
husein-MS-7D31:1463326:1463519 [0] NCCL INFO [Proxy Service] Device 0 CPU core 4
husein-MS-7D31:1463326:1463521 [0] NCCL INFO [Proxy Service UDS] Device 0 CPU core 14
husein-MS-7D31:1463327:1463520 [1] NCCL INFO [Proxy Service UDS] Device 1 CPU core 6
husein-MS-7D31:1463327:1463515 [1] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
husein-MS-7D31:1463327:1463515 [1] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
husein-MS-7D31:1463326:1463512 [0] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
husein-MS-7D31:1463326:1463512 [0] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
husein-MS-7D31:1463326:1463512 [0] NCCL INFO CC Off, workFifoBytes 1048576
husein-MS-7D31:1463326:1463512 [0] NCCL INFO TUNER/Plugin: Could not find: libnccl-tuner.so. Using internal tuner plugin.
husein-MS-7D31:1463327:1463515 [1] NCCL INFO TUNER/Plugin: Could not find: libnccl-tuner.so. Using internal tuner plugin.
husein-MS-7D31:1463326:1463512 [0] NCCL INFO ncclCommInitRankConfig comm 0x4452af40 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId 1000 commId 0x798dcd8b6f24bb47 - Init COMPLETE
husein-MS-7D31:1463327:1463515 [1] NCCL INFO ncclCommInitRankConfig comm 0x35e5f030 rank 1 nranks 2 cudaDev 1 nvmlDev 1 busId 8000 commId 0x798dcd8b6f24bb47 - Init COMPLETE
husein-MS-7D31:1463326:1463512 [0] NCCL INFO Init timings - ncclCommInitRankConfig: rank 0 nranks 2 total 0.91 (kernels 0.10, alloc 0.00, bootstrap 0.79, allgathers 0.00, topo 0.01, graphs 0.00, connections 0.00, rest 0.00)
husein-MS-7D31:1463327:1463515 [1] NCCL INFO Init timings - ncclCommInitRankConfig: rank 1 nranks 2 total 0.09 (kernels 0.07, alloc 0.00, bootstrap 0.00, allgathers 0.00, topo 0.01, graphs 0.00, connections 0.00, rest 0.00)
husein-MS-7D31:1463326:1463523 [0] NCCL INFO Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
husein-MS-7D31:1463327:1463522 [1] NCCL INFO Channel 00 : 1[1] -> 0[0] via SHM/direct/direct
husein-MS-7D31:1463326:1463523 [0] NCCL INFO Channel 01 : 0[0] -> 1[1] via SHM/direct/direct
husein-MS-7D31:1463327:1463522 [1] NCCL INFO Channel 01 : 1[1] -> 0[0] via SHM/direct/direct
husein-MS-7D31:1463326:1463523 [0] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
husein-MS-7D31:1463327:1463522 [1] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
  0%|                                                   | 0/279 [00:00<?, ?it/s][rank0]:[W130 13:08:42.490301705 reducer.cpp:1431] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[rank1]:[W130 13:08:42.490301711 reducer.cpp:1431] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
{'train_runtime': 4.8779, 'train_samples_per_second': 32.801, 'train_steps_per_second': 2.05, 'train_loss': 0.5222671627998352, 'epoch': 0.04}
100%|███████████████████████████████████████████| 10/10 [00:04<00:00,  2.10it/s]
husein-MS-7D31:176395:176395 [1] NCCL INFO comm 0x3bcb2f40 rank 1 nranks 2 cudaDev 1 busId 8000 - Destroy COMPLETE
husein-MS-7D31:176394:176394 [0] NCCL INFO comm 0x10233d80 rank 0 nranks 2 cudaDev 0 busId 1000 - Destroy COMPLETE
Deleted virtual environment at /home/ubuntu/.venv-3.12-v2
```

Checkout [examples/simple_ddp.py](examples/simple_ddp.py)

### Multi one time jobs

```python
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
```

Output,

```
Using CPython 3.11.13 interpreter at: /usr/bin/python3.11
Creating virtual environment at: .venv-3.11
Activate with: source .venv-3.11/bin/activate
Using CPython 3.9.16 interpreter at: /usr/bin/python3.9
Creating virtual environment at: .venv-3.9
Activate with: source .venv-3.9/bin/activate
Using CPython 3.13.11
Creating virtual environment at: .venv-3.13
Activate with: source .venv-3.13/bin/activate
Using CPython 3.12.12
Creating virtual environment at: .venv-3.12
Activate with: source .venv-3.12/bin/activate
Using CPython 3.10.17 interpreter at: /usr/bin/python3.10
Creating virtual environment at: .venv-3.10
Activate with: source .venv-3.10/bin/activate
Using Python 3.9.16 environment at: .venv-3.9
Using Python 3.11.13 environment at: .venv-3.11
Using Python 3.13.11 environment at: .venv-3.13
Using Python 3.12.12 environment at: .venv-3.12
Using Python 3.10.17 environment at: .venv-3.10
Resolved 1 package in 47ms
Installed 1 package in 1ms
 + cloudpickle==3.1.2
Resolved 1 package in 57ms
Installed 1 package in 1ms
 + cloudpickle==3.1.2
Resolved 1 package in 55ms
Installed 1 package in 1ms
 + cloudpickle==3.1.2
Resolved 1 package in 68ms
Installed 1 package in 1ms
 + cloudpickle==3.1.2
Resolved 1 package in 76ms
Installed 1 package in 1ms
 + cloudpickle==3.1.2
Using Python 3.9.16 environment at: .venv-3.9
Resolved 6 packages in 4ms
Using Python 3.12.12 environment at: .venv-3.12
Using Python 3.13.11 environment at: .venv-3.13
Installed 6 packages in 9ms
 + numpy==1.26.4
 + pandas==2.2.3
 + python-dateutil==2.9.0.post0
 + pytz==2025.2
 + six==1.17.0
 + tzdata==2025.3
Resolved 6 packages in 4ms
Resolved 6 packages in 4ms
Installed 6 packages in 9ms
 + numpy==1.26.4
 + pandas==2.2.3
 + python-dateutil==2.9.0.post0
 + pytz==2025.2
 + six==1.17.0
 + tzdata==2025.3
Using Python 3.10.17 environment at: .venv-3.10
Using Python 3.11.13 environment at: .venv-3.11
Installed 6 packages in 10ms
 + numpy==1.26.4
 + pandas==2.2.3
 + python-dateutil==2.9.0.post0
 + pytz==2025.2
 + six==1.17.0
 + tzdata==2025.3
Resolved 6 packages in 4ms
Resolved 6 packages in 4ms
Installed 6 packages in 9ms
 + numpy==1.26.4
 + pandas==2.2.3
 + python-dateutil==2.9.0.post0
 + pytz==2025.2
 + six==1.17.0
 + tzdata==2025.3
Installed 6 packages in 13ms
 + numpy==1.26.4
 + pandas==2.2.3
 + python-dateutil==2.9.0.post0
 + pytz==2025.2
 + six==1.17.0
 + tzdata==2025.3
inside compute() 3.10.17 (main, Apr  9 2025, 08:54:15) [GCC 9.4.0]
inside compute() 3.9.16 (main, Dec  7 2022, 01:11:51) 
[GCC 9.4.0]
inside compute() 3.11.13 (main, Jun  4 2025, 08:57:30) [GCC 9.4.0]
inside compute() 3.13.11 (main, Jan 14 2026, 19:38:04) [Clang 21.1.4 ]
inside compute() 3.12.12 (main, Dec  9 2025, 19:02:36) [Clang 21.1.4 ]
Deleted virtual environment at /home/ubuntu/.venv-3.9
Deleted virtual environment at /home/ubuntu/.venv-3.11
Deleted virtual environment at /home/ubuntu/.venv-3.10
Deleted virtual environment at /home/ubuntu/.venv-3.12
Deleted virtual environment at /home/ubuntu/.venv-3.13
```

Checkout [examples/simple_onetime_mp.py](examples/simple_onetime_mp.py)

### Jupyter Notebook

```python
from pyremote import remote, remotecell, UvConfig, close_remote
import numpy as np
```

```python
remote(
    "localhost", 
    "ubuntu", 
    password="ubuntu123", 
    uv=UvConfig(path="~/.venv-3.11", python_version="3.11", install_uv=True, delete_after_done=True), 
    dependencies=["numpy==1.26.4", "pandas==2.2.3", "torch==2.9.1"],
    install_verbose=True,
    jupyter_mode=True,
)
```

```
    Using CPython 3.11.13 interpreter at: /usr/bin/python3.11
    Creating virtual environment at: .venv-3.11
    Activate with: source .venv-3.11/bin/activate
    Using Python 3.11.13 environment at: .venv-3.11
    Resolved 1 package in 0.96ms
    Installed 1 package in 2ms
     + cloudpickle==3.1.2
    Using Python 3.11.13 environment at: .venv-3.11
    Resolved 31 packages in 8ms
    Installed 31 packages in 90ms
     + filelock==3.20.3
     + fsspec==2026.1.0
     + jinja2==3.1.6
     + markupsafe==3.0.3
     + mpmath==1.3.0
     + networkx==3.6.1
     + numpy==1.26.4
     + nvidia-cublas-cu12==12.8.4.1
     + nvidia-cuda-cupti-cu12==12.8.90
     + nvidia-cuda-nvrtc-cu12==12.8.93
     + nvidia-cuda-runtime-cu12==12.8.90
     + nvidia-cudnn-cu12==9.10.2.21
     + nvidia-cufft-cu12==11.3.3.83
     + nvidia-cufile-cu12==1.13.1.3
     + nvidia-curand-cu12==10.3.9.90
     + nvidia-cusolver-cu12==11.7.3.90
     + nvidia-cusparse-cu12==12.5.8.93
     + nvidia-cusparselt-cu12==0.7.1
     + nvidia-nccl-cu12==2.27.5
     + nvidia-nvjitlink-cu12==12.8.93
     + nvidia-nvshmem-cu12==3.3.20
     + nvidia-nvtx-cu12==12.8.90
     + pandas==2.2.3
     + python-dateutil==2.9.0.post0
     + pytz==2025.2
     + six==1.17.0
     + sympy==1.14.0
     + torch==2.9.1
     + triton==3.5.1
     + typing-extensions==4.15.0
     + tzdata==2025.3
```

```python
remote(
    "localhost", 
    "ubuntu", 
    password="ubuntu123", 
    uv=UvConfig(path="~/.venv-3.10", python_version="3.10", install_uv=True, delete_after_done=True), 
    dependencies=["numpy==1.26.4", "pandas==2.2.3", "torch==2.9.1"],
    install_verbose=True,
    jupyter_mode=True,
    jupyter_profile="3.10"
)
```

```
    Using CPython 3.10.17 interpreter at: /usr/bin/python3.10
    Creating virtual environment at: .venv-3.10
    Activate with: source .venv-3.10/bin/activate
    Using Python 3.10.17 environment at: .venv-3.10
    Resolved 1 package in 1ms
    Installed 1 package in 1ms
     + cloudpickle==3.1.2
    Using Python 3.10.17 environment at: .venv-3.10
    Resolved 31 packages in 7ms
    Installed 31 packages in 71ms
     + filelock==3.20.3
     + fsspec==2026.1.0
     + jinja2==3.1.6
     + markupsafe==3.0.3
     + mpmath==1.3.0
     + networkx==3.4.2
     + numpy==1.26.4
     + nvidia-cublas-cu12==12.8.4.1
     + nvidia-cuda-cupti-cu12==12.8.90
     + nvidia-cuda-nvrtc-cu12==12.8.93
     + nvidia-cuda-runtime-cu12==12.8.90
     + nvidia-cudnn-cu12==9.10.2.21
     + nvidia-cufft-cu12==11.3.3.83
     + nvidia-cufile-cu12==1.13.1.3
     + nvidia-curand-cu12==10.3.9.90
     + nvidia-cusolver-cu12==11.7.3.90
     + nvidia-cusparse-cu12==12.5.8.93
     + nvidia-cusparselt-cu12==0.7.1
     + nvidia-nccl-cu12==2.27.5
     + nvidia-nvjitlink-cu12==12.8.93
     + nvidia-nvshmem-cu12==3.3.20
     + nvidia-nvtx-cu12==12.8.90
     + pandas==2.2.3
     + python-dateutil==2.9.0.post0
     + pytz==2025.2
     + six==1.17.0
     + sympy==1.14.0
     + torch==2.9.1
     + triton==3.5.1
     + typing-extensions==4.15.0
     + tzdata==2025.3
```

```python
x = 10
result = np.array([1, 2, 3])
```

```python
%%remotecell
import numpy as np
import torch
import sys

print(sys.version)

y = np.array([1, 2, 3])
print(f"Defined x={x}, result={result}")

t = torch.randn(10, 10).cuda()
print(t)
```

```
    3.11.13 (main, Jun  4 2025, 08:57:30) [GCC 9.4.0]
    Defined x=10, result=[1 2 3]
    tensor([[ 6.1146e-01,  2.2552e-01, -7.4423e-01,  2.1329e-01,  1.7878e+00,
              1.6502e+00,  6.1555e-01,  9.6762e-01, -9.2587e-02,  8.1585e-01],
            [ 1.4931e+00, -6.8062e-01, -8.9354e-01,  5.4590e-01,  1.9131e-03,
              2.3486e-02, -1.8477e-01,  4.7742e-02, -6.4516e-01, -1.1166e-01],
            [ 5.0783e-01,  9.5421e-02,  1.0320e+00,  1.0705e+00, -2.7514e-01,
             -2.7833e-01, -1.8243e-01, -2.2620e-01,  1.8950e+00, -4.2753e-01],
            [ 1.2461e+00, -1.3281e+00, -2.1273e+00,  1.2477e+00,  5.9034e-01,
             -9.1703e-01,  3.9618e-01,  2.7257e-01,  3.9846e-01, -2.2619e+00],
            [ 1.8078e+00,  1.0062e+00, -8.1295e-02, -4.7849e-01,  2.8566e-02,
              1.9351e-01,  1.9739e+00,  9.0229e-03,  2.6468e-01,  7.7196e-01],
            [-3.6440e-01, -1.3292e+00,  5.7839e-01, -5.5599e-01, -6.3926e-01,
              1.2966e+00, -7.8355e-01,  4.3229e-01, -6.4102e-01, -8.6305e-02],
            [-8.2536e-01, -1.0366e+00, -3.6685e-01, -1.0335e+00,  6.1879e-01,
             -1.2390e+00, -6.1706e-01, -1.2072e+00,  2.4161e+00, -9.8014e-01],
            [-1.3233e+00, -1.4276e+00, -1.8212e+00, -9.6309e-01,  7.0925e-01,
              2.8101e-01, -3.8862e-01, -4.3115e-01,  3.3140e-01, -9.2514e-02],
            [-6.6766e-01,  2.3285e-01,  4.4315e-01, -1.7091e-01,  7.4387e-01,
             -3.2802e-01, -1.1730e+00, -6.1005e-01,  9.1896e-01, -1.6952e+00],
            [ 4.0766e-02, -9.2963e-01, -7.0721e-01, -9.4238e-01, -5.6975e-01,
              8.7586e-01, -1.0832e+00,  5.1352e-01,  1.2193e+00,  5.9838e-01]],
           device='cuda:0')
```

```python
%%remotecell --profile 3.10

print(sys.version)
print(t)
```

```
    3.10.17 (main, Apr  9 2025, 08:54:15) [GCC 9.4.0]
    tensor([[ 6.1146e-01,  2.2552e-01, -7.4423e-01,  2.1329e-01,  1.7878e+00,
              1.6502e+00,  6.1555e-01,  9.6762e-01, -9.2587e-02,  8.1585e-01],
            [ 1.4931e+00, -6.8062e-01, -8.9354e-01,  5.4590e-01,  1.9131e-03,
              2.3486e-02, -1.8477e-01,  4.7742e-02, -6.4516e-01, -1.1166e-01],
            [ 5.0783e-01,  9.5421e-02,  1.0320e+00,  1.0705e+00, -2.7514e-01,
             -2.7833e-01, -1.8243e-01, -2.2620e-01,  1.8950e+00, -4.2753e-01],
            [ 1.2461e+00, -1.3281e+00, -2.1273e+00,  1.2477e+00,  5.9034e-01,
             -9.1703e-01,  3.9618e-01,  2.7257e-01,  3.9846e-01, -2.2619e+00],
            [ 1.8078e+00,  1.0062e+00, -8.1295e-02, -4.7849e-01,  2.8566e-02,
              1.9351e-01,  1.9739e+00,  9.0229e-03,  2.6468e-01,  7.7196e-01],
            [-3.6440e-01, -1.3292e+00,  5.7839e-01, -5.5599e-01, -6.3926e-01,
              1.2966e+00, -7.8355e-01,  4.3229e-01, -6.4102e-01, -8.6305e-02],
            [-8.2536e-01, -1.0366e+00, -3.6685e-01, -1.0335e+00,  6.1879e-01,
             -1.2390e+00, -6.1706e-01, -1.2072e+00,  2.4161e+00, -9.8014e-01],
            [-1.3233e+00, -1.4276e+00, -1.8212e+00, -9.6309e-01,  7.0925e-01,
              2.8101e-01, -3.8862e-01, -4.3115e-01,  3.3140e-01, -9.2514e-02],
            [-6.6766e-01,  2.3285e-01,  4.4315e-01, -1.7091e-01,  7.4387e-01,
             -3.2802e-01, -1.1730e+00, -6.1005e-01,  9.1896e-01, -1.6952e+00],
            [ 4.0766e-02, -9.2963e-01, -7.0721e-01, -9.4238e-01, -5.6975e-01,
              8.7586e-01, -1.0832e+00,  5.1352e-01,  1.2193e+00,  5.9838e-01]],
           device='cuda:0')
```

```python
%%remotecell

global result

result = np.concatenate([result, result])
print(result)
```

```
    [1 2 3 1 2 3]
```

```python
close_remote()
```

```
    Deleted virtual environment at /home/ubuntu/.venv-3.11
    Remote connection closed.
    Deleted virtual environment at /home/ubuntu/.venv-3.10
    Remote connection closed.
```

Checkout [examples/simple_notebook.ipynb](examples/simple_notebook.ipynb)