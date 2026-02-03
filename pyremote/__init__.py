import ast
import inspect
import textwrap
import cloudpickle
import base64
import hashlib
import os
from typing import Any, List, Optional, Dict, Set, Union, Callable, Literal
from dataclasses import dataclass, field
from functools import wraps
import paramiko
import sys
import ctypes
import dis


class RemoteExecutionError(Exception):
    """Base exception for remote execution errors."""
    pass


class RemoteImportError(RemoteExecutionError):
    """Failed to install dependencies on remote."""
    pass


class RemoteConnectionError(RemoteExecutionError):
    """Failed to connect to remote host."""
    pass


class RemoteVenvError(RemoteExecutionError):
    """Failed to create or activate virtual environment."""
    pass


@dataclass
class SSHConfig:
    host: str
    username: str
    port: int = 22
    password: Optional[str] = None
    key_filename: Optional[str] = None
    key_password: Optional[str] = None
    timeout: int = 30
    env: Optional[Dict[str, str]] = None  # Environment variables


@dataclass
class VenvConfig:
    """Standard venv configuration."""
    path: str
    create_if_missing: bool = True
    python_base: str = "python3"
    delete_after_done: bool = False
    
    @property
    def python_path(self) -> str:
        return f"{self.path}/bin/python3"
    
    @property
    def pip_path(self) -> str:
        return f"{self.path}/bin/pip"
    
    @property
    def activate_path(self) -> str:
        return f"{self.path}/bin/activate"


@dataclass
class UvConfig:
    """UV-based virtual environment configuration."""
    path: str = ".venv"
    python_version: Optional[str] = None
    create_if_missing: bool = True
    install_uv: bool = True
    delete_after_done: bool = False
    
    @property
    def python_path(self) -> str:
        return f"{self.path}/bin/python3"
    
    @property
    def pip_path(self) -> str:
        return f"{self.path}/bin/pip"
    
    @property
    def activate_path(self) -> str:
        return f"{self.path}/bin/activate"


@dataclass
class MultiprocessingConfig:
    """Configuration for multi-GPU/multi-process distributed execution.
    
    Args:
        num_processes: Number of processes to spawn. Use "auto" to detect GPU count,
                      or an integer for explicit count.
        master_addr: Address for distributed communication (default: localhost)
        master_port: Port for distributed communication (default: 29500)
        backend: Distributed backend - "nccl" for GPU, "gloo" for CPU (default: nccl)
        start_method: Multiprocessing start method - "spawn", "fork", or "forkserver"
    """
    num_processes: Union[int, Literal["auto"]] = "auto"
    master_addr: str = "localhost"
    master_port: int = 29500
    backend: str = "nccl"
    start_method: str = "spawn"


@dataclass 
class RemoteResult:
    stdout: str
    stderr: str
    return_value: Any
    modified_vars: Dict[str, Any]
    exception: Optional[Exception] = None


class RemoteExecutor:
    def __init__(
        self,
        ssh_config: SSHConfig,
        venv: Optional[Union[str, VenvConfig]] = None,
        uv: Optional[Union[str, UvConfig]] = None,
        dependencies: List[str] = None,
        python_path: Optional[str] = None,
        setup_commands: List[str] = None,
        install_verbose: bool = False,
        stdout_callback: Optional[Callable[[str], None]] = None,
        multiprocessing: Optional[Union[int, Literal["auto"], MultiprocessingConfig]] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        self.ssh_config = ssh_config
        self.dependencies = dependencies or []
        self.setup_commands = setup_commands or []
        self.install_verbose = install_verbose
        self.stdout_callback = stdout_callback
        self.env = env or {}
        
        # Handle multiprocessing config
        if multiprocessing is None:
            self.mp_config = None
        elif isinstance(multiprocessing, int):
            self.mp_config = MultiprocessingConfig(num_processes=multiprocessing)
        elif multiprocessing == "auto":
            self.mp_config = MultiprocessingConfig(num_processes="auto")
        elif isinstance(multiprocessing, MultiprocessingConfig):
            self.mp_config = multiprocessing
        else:
            self.mp_config = None
        
        if isinstance(venv, str):
            self.venv = VenvConfig(path=venv)
        elif isinstance(venv, VenvConfig):
            self.venv = venv
        else:
            self.venv = None
        
        if isinstance(uv, str):
            self.uv = UvConfig(path=uv)
        elif isinstance(uv, UvConfig):
            self.uv = uv
        else:
            self.uv = None
        
        if self.uv and self.venv:
            self.venv = None
        
        if python_path:
            self._python_path = python_path
        elif self.uv:
            self._python_path = self.uv.python_path
        elif self.venv:
            self._python_path = self.venv.python_path
        else:
            self._python_path = "python3"
        
        self._client: Optional[paramiko.SSHClient] = None
        self._deps_installed: Set[str] = set()
        self._venv_verified: bool = False
        self._uv_verified: bool = False
        self._expanded_venv_path: Optional[str] = None  # Track expanded path for cleanup
        self._expanded_uv_path: Optional[str] = None  # Track expanded path for cleanup
    
    def _connect(self):
        try:
            self._client = paramiko.SSHClient()
            self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            connect_kwargs = {
                'hostname': self.ssh_config.host,
                'port': self.ssh_config.port,
                'username': self.ssh_config.username,
                'timeout': self.ssh_config.timeout,
            }
            
            if self.ssh_config.key_filename:
                key_path = os.path.expanduser(self.ssh_config.key_filename)
                connect_kwargs['key_filename'] = key_path
                if self.ssh_config.key_password:
                    connect_kwargs['passphrase'] = self.ssh_config.key_password
            elif self.ssh_config.password:
                connect_kwargs['password'] = self.ssh_config.password
            
            self._client.connect(**connect_kwargs)
        except paramiko.AuthenticationException as e:
            raise RemoteConnectionError(f"Authentication failed: {e}") from e
        except paramiko.SSHException as e:
            raise RemoteConnectionError(f"SSH connection failed: {e}") from e
        except Exception as e:
            raise RemoteConnectionError(f"Connection failed: {e}") from e
    
    def _disconnect(self):
        if self._client:
            self._client.close()
            self._client = None
    
    def _run_command(self, cmd: str, timeout: int = None, stream: bool = False) -> tuple:
        """Run a command on remote and return exit_status, stdout, stderr."""
        timeout = timeout or self.ssh_config.timeout
        
        if '\n' not in cmd.strip():
            wrapped_cmd = f"bash -l -c '{cmd}'"
        else:
            cmd_encoded = base64.b64encode(cmd.encode()).decode()
            wrapped_cmd = f'echo {cmd_encoded} | base64 -d | bash -l'
        
        stdin, stdout, stderr = self._client.exec_command(wrapped_cmd, timeout=timeout)
        
        if stream:
            import time
            channel = stdout.channel
            output_lines = []
            error_lines = []
            stdout_buffer = ""
            stderr_buffer = ""
            
            while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
                if channel.recv_ready():
                    chunk = channel.recv(4096).decode('utf-8', errors='replace')
                    stdout_buffer += chunk
                    
                    while '\n' in stdout_buffer:
                        line, stdout_buffer = stdout_buffer.split('\n', 1)
                        output_lines.append(line)
                        print(line, flush=True)
                
                if channel.recv_stderr_ready():
                    chunk = channel.recv_stderr(4096).decode('utf-8', errors='replace')
                    stderr_buffer += chunk
                    
                    while '\n' in stderr_buffer:
                        line, stderr_buffer = stderr_buffer.split('\n', 1)
                        error_lines.append(line)
                        print(line, file=sys.stderr, flush=True)
                
                if not channel.recv_ready() and not channel.recv_stderr_ready():
                    time.sleep(0.01)
            
            if stdout_buffer:
                output_lines.append(stdout_buffer)
                print(stdout_buffer, flush=True)
            if stderr_buffer:
                error_lines.append(stderr_buffer)
                print(stderr_buffer, file=sys.stderr, flush=True)
            
            exit_status = channel.recv_exit_status()
            stdout_content = '\n'.join(output_lines)
            stderr_content = '\n'.join(error_lines)
        else:
            exit_status = stdout.channel.recv_exit_status()
            stdout_content = stdout.read().decode()
            stderr_content = stderr.read().decode()
        
        return exit_status, stdout_content, stderr_content
    
    def _setup_uv(self):
        if not self.uv:
            return
        
        if self._uv_verified:
            return
        
        uv_path = self.uv.path
        
        exit_status, expanded_path, _ = self._run_command(f'echo {uv_path}')
        if exit_status == 0:
            uv_path = expanded_path.strip()
            self._expanded_uv_path = uv_path  # Store expanded path for cleanup
            self.uv = UvConfig(
                path=uv_path,
                python_version=self.uv.python_version,
                create_if_missing=self.uv.create_if_missing,
                install_uv=self.uv.install_uv,
                delete_after_done=self.uv.delete_after_done,
            )
            self._python_path = self.uv.python_path
        
        exit_status, out, _ = self._run_command('command -v uv')
        
        if exit_status != 0:
            if not self.uv.install_uv:
                raise RemoteVenvError(
                    "UV is not installed on remote and install_uv=False"
                )
            
            install_cmd = 'curl -LsSf https://astral.sh/uv/install.sh | sh'
            exit_status, stdout, stderr = self._run_command(
                install_cmd, timeout=120, stream=self.install_verbose
            )
            
            if exit_status != 0:
                raise RemoteVenvError(f"Failed to install UV: {stderr}")
        
        exit_status, _, _ = self._run_command(f'test -f {self.uv.python_path}')
        
        if exit_status != 0:
            if not self.uv.create_if_missing:
                raise RemoteVenvError(
                    f"Virtual environment not found at {uv_path} "
                    f"and create_if_missing=False"
                )
            
            python_opt = f'--python {self.uv.python_version}' if self.uv.python_version else ''
            create_cmd = f'''
                source $HOME/.local/bin/env 2>/dev/null || true
                uv venv {python_opt} --allow-existing {uv_path}
            '''
            exit_status, stdout, stderr = self._run_command(
                create_cmd, timeout=180, stream=self.install_verbose
            )
            
            if exit_status != 0:
                raise RemoteVenvError(
                    f"Failed to create UV virtual environment at {uv_path}: {stderr}"
                )
            
            pip_cmd = f'''
                source $HOME/.local/bin/env 2>/dev/null || true
                source {self.uv.activate_path}
                uv pip install cloudpickle
            '''
            exit_status, stdout, stderr = self._run_command(
                pip_cmd, timeout=120, stream=self.install_verbose
            )
            
            if exit_status != 0:
                raise RemoteVenvError(
                    f"Failed to install cloudpickle in UV venv: {stderr}"
                )
        
        self._uv_verified = True
    
    def _setup_venv(self):
        if not self.venv:
            return
        
        if self._venv_verified:
            return
        
        venv_path = self.venv.path
        
        exit_status, expanded_path, _ = self._run_command(f'echo {venv_path}')
        if exit_status == 0:
            venv_path = expanded_path.strip()
            self._expanded_venv_path = venv_path  # Store expanded path for cleanup
            self.venv = VenvConfig(
                path=venv_path,
                create_if_missing=self.venv.create_if_missing,
                python_base=self.venv.python_base,
                delete_after_done=self.venv.delete_after_done,
            )
            self._python_path = self.venv.python_path
        
        exit_status, _, _ = self._run_command(f'test -f {self.venv.python_path}')
        
        if exit_status != 0:
            if not self.venv.create_if_missing:
                raise RemoteVenvError(
                    f"Virtual environment not found at {venv_path} "
                    f"and create_if_missing=False"
                )
            
            create_cmd = f'{self.venv.python_base} -m venv {venv_path}'
            exit_status, stdout, stderr = self._run_command(
                create_cmd, timeout=120, stream=self.install_verbose
            )
            
            if exit_status != 0:
                raise RemoteVenvError(
                    f"Failed to create virtual environment at {venv_path}: {stderr}"
                )
            
            pip_cmd = f'{self.venv.pip_path} install cloudpickle'
            exit_status, stdout, stderr = self._run_command(
                pip_cmd, timeout=120, stream=self.install_verbose
            )
            
            if exit_status != 0:
                raise RemoteVenvError(
                    f"Failed to install cloudpickle in venv: {stderr}"
                )
        
        self._venv_verified = True
    
    def _cleanup_venv(self):
        """Delete the virtual environment if delete_after_done is True."""
        path_to_delete = None
        
        if self.uv and self.uv.delete_after_done:
            path_to_delete = self._expanded_uv_path or self.uv.path
        elif self.venv and self.venv.delete_after_done:
            path_to_delete = self._expanded_venv_path or self.venv.path
        
        if path_to_delete:
            # Safety check: ensure path is not empty or root-like
            if path_to_delete and path_to_delete not in ('/', '/home', '/root', '~'):
                delete_cmd = f'rm -rf {path_to_delete}'
                exit_status, stdout, stderr = self._run_command(delete_cmd, timeout=60, stream=self.install_verbose)
                if exit_status != 0:
                    print(f"Warning: Failed to delete virtual environment at {path_to_delete}: {stderr}", 
                          file=sys.stderr)
                elif self.install_verbose:
                    print(f"Deleted virtual environment at {path_to_delete}")
    
    def _install_dependencies(self):
        if not self.dependencies:
            return
        
        deps_key = ','.join(sorted(self.dependencies)) + str(self.venv) + str(self.uv)
        deps_hash = hashlib.md5(deps_key.encode()).hexdigest()[:8]
        
        if deps_hash in self._deps_installed:
            return
        
        deps_str = ' '.join(f'"{dep}"' for dep in self.dependencies)
        
        if self.uv:
            cmd = f'''
                source $HOME/.local/bin/env 2>/dev/null || true
                source {self.uv.activate_path}
                uv pip install {deps_str}
            '''
        elif self.venv:
            cmd = f'{self.venv.pip_path} install {deps_str}'
        else:
            cmd = f'{self._python_path} -m pip install {deps_str}'
        
        exit_status, stdout, stderr = self._run_command(
            cmd, timeout=300, stream=self.install_verbose
        )
        
        if exit_status != 0:
            raise RemoteImportError(
                f"Failed to install dependencies {self.dependencies}: {stderr}"
            )
        
        self._deps_installed.add(deps_hash)
    
    def _run_setup_commands(self):
        for cmd in self.setup_commands:
            exit_status, stdout, stderr = self._run_command(cmd)
            if exit_status != 0:
                raise RemoteExecutionError(
                    f"Setup command failed: {cmd}\nstderr: {stderr}"
                )
    
    def _extract_captured_vars(self, func: Callable, frame) -> Dict[str, Any]:
        captured = {}
        code = func.__code__
        
        if code.co_freevars and func.__closure__:
            for i, var in enumerate(code.co_freevars):
                try:
                    val = func.__closure__[i].cell_contents
                    cloudpickle.dumps(val)
                    captured[var] = val
                except (ValueError, Exception):
                    pass
        
        global_vars = set()
        for instr in dis.get_instructions(code):
            if instr.opname in ('LOAD_GLOBAL', 'STORE_GLOBAL'):
                global_vars.add(instr.argval)
        
        for var in global_vars:
            if var in captured:
                continue
            
            if var in frame.f_locals:
                try:
                    val = frame.f_locals[var]
                    cloudpickle.dumps(val)
                    captured[var] = val
                except Exception:
                    pass
            elif var in frame.f_globals:
                try:
                    val = frame.f_globals[var]
                    if not inspect.ismodule(val) and not inspect.isbuiltin(val):
                        cloudpickle.dumps(val)
                        captured[var] = val
                except Exception:
                    pass
        
        return captured
    
    def _get_function_source(self, func: Callable) -> Optional[str]:
        """Extract function source code, removing decorators."""
        try:
            source = inspect.getsource(func)
            lines = source.split('\n')
            
            func_start = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('def '):
                    func_start = i
                    break
            
            func_lines = lines[func_start:]
            
            if func_lines:
                min_indent = float('inf')
                for line in func_lines:
                    if line.strip():
                        indent = len(line) - len(line.lstrip())
                        min_indent = min(min_indent, indent)
                
                if min_indent < float('inf') and min_indent > 0:
                    func_lines = [
                        line[min_indent:] if len(line) >= min_indent else line 
                        for line in func_lines
                    ]
            
            return '\n'.join(func_lines)
        except (OSError, TypeError):
            return None
    
    def _execute_remote(self, func: Callable, args: tuple, kwargs: dict, 
                        captured_vars: Dict[str, Any]) -> RemoteResult:
        
        func_source = self._get_function_source(func)
        
        # Convert mp_config to plain dict to avoid pickle issues on remote
        mp_config_dict = None
        if self.mp_config is not None:
            mp_config_dict = {
                'num_processes': self.mp_config.num_processes,
                'master_addr': self.mp_config.master_addr,
                'master_port': self.mp_config.master_port,
                'backend': self.mp_config.backend,
                'start_method': self.mp_config.start_method,
            }
        
        payload = {
            'func_source': func_source,
            'func_name': func.__name__,
            'args': args,
            'kwargs': kwargs,
            'captured_vars': captured_vars,
            'mp_config': mp_config_dict,
            'env': self.env,
        }
        
        if func_source is None:
            payload['func'] = func
        
        payload_encoded = base64.b64encode(cloudpickle.dumps(payload)).decode()
        
        executor_script = self._build_executor_script()
        
        if self.uv:
            activate_cmd = f'source $HOME/.local/bin/env 2>/dev/null || true && source {self.uv.activate_path}'
            python_cmd = 'python3'
        elif self.venv:
            activate_cmd = f'source {self.venv.activate_path}'
            python_cmd = 'python3'
        else:
            activate_cmd = ''
            python_cmd = self._python_path
        
        # For multiprocessing, we need to write the script to a file
        # so that spawn can properly import worker_fn
        if self.mp_config is not None:
            # Write script to temp file on remote
            script_b64 = base64.b64encode(executor_script.encode()).decode()
            
            write_script_cmd = f'''
echo "{script_b64}" | base64 -d > /tmp/_pyremote_executor.py
'''
            exit_status, _, stderr = self._run_command(write_script_cmd)
            if exit_status != 0:
                raise RemoteExecutionError(f"Failed to write executor script: {stderr}")
            
            if activate_cmd:
                remote_cmd = f'''bash -c '
{activate_cmd} && {python_cmd} /tmp/_pyremote_executor.py "{payload_encoded}"
'
'''
            else:
                remote_cmd = f'''bash -c '
{python_cmd} /tmp/_pyremote_executor.py "{payload_encoded}"
'
'''
        else:
            # For non-multiprocessing, we can use exec() as before
            script_encoded = base64.b64encode(executor_script.encode()).decode()
            
            if activate_cmd:
                remote_cmd = f'''bash -c '
{activate_cmd} && {python_cmd} -c "
import base64, sys
script = base64.b64decode(\\"{script_encoded}\\").decode()
exec(script)
" "{payload_encoded}"
'
'''
            else:
                remote_cmd = f'''bash -c '
{python_cmd} -c "
import base64, sys
script = base64.b64decode(\\"{script_encoded}\\").decode()
exec(script)
" "{payload_encoded}"
'
'''
        
        stdin, stdout, stderr = self._client.exec_command(
            remote_cmd, 
            timeout=self.ssh_config.timeout * 10,
            get_pty=True,
        )
        
        channel = stdout.channel
        result_data = None
        output_lines = []
        buffer = ""
        
        while not channel.exit_status_ready() or channel.recv_ready():
            if channel.recv_ready():
                chunk = channel.recv(4096).decode('utf-8', errors='replace')
                buffer += chunk
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.startswith('__REMOTE_RESULT__:'):
                        result_encoded = line.split(':', 1)[1]
                        try:
                            result_data = cloudpickle.loads(base64.b64decode(result_encoded))
                        except Exception:
                            pass
                    else:
                        output_lines.append(line)
                        print(line, flush=True)
                        if self.stdout_callback:
                            self.stdout_callback(line)
            else:
                import time
                time.sleep(0.01)
        
        if buffer:
            if buffer.startswith('__REMOTE_RESULT__:'):
                result_encoded = buffer.split(':', 1)[1]
                try:
                    result_data = cloudpickle.loads(base64.b64decode(result_encoded))
                except Exception:
                    pass
            else:
                output_lines.append(buffer)
                print(buffer, flush=True)
                if self.stdout_callback:
                    self.stdout_callback(buffer)
        
        stderr_content = stderr.read().decode()
        actual_stdout = '\n'.join(output_lines)
        if output_lines:
            actual_stdout += '\n'
        
        if result_data is None:
            return RemoteResult(
                stdout=actual_stdout,
                stderr=stderr_content,
                return_value=None,
                modified_vars={},
                exception=RemoteExecutionError(
                    f"Failed to get result from remote.\n"
                    f"stdout: {actual_stdout}\n"
                    f"stderr: {stderr_content}"
                )
            )
        
        return RemoteResult(
            stdout=actual_stdout,
            stderr=stderr_content,
            return_value=result_data.get('return_value'),
            modified_vars=result_data.get('modified_vars', {}),
            exception=result_data.get('exception'),
        )
    
    def _build_executor_script(self) -> str:
        return '''#!/usr/bin/env python3
import sys
import base64
import traceback
import os
import types

def worker_fn(rank, world_size, func_source, func_name, args, kwargs, captured_vars, result_queue, backend, env_vars):
    """Worker function that runs on each process."""
    import torch
    import torch.distributed as dist
    
    try:
        # Set environment variables (need to do this in each worker)
        for key, value in env_vars.items():
            os.environ[key] = str(value)
        
        # Initialize distributed
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
        
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
        
        # Recreate the function in this process
        exec_globals = {'__builtins__': __builtins__}
        exec_globals.update(captured_vars)
        exec(func_source, exec_globals)
        user_func = exec_globals[func_name]
        
        # Call user function WITHOUT injecting rank/world_size
        # User can access via:
        #   - os.environ['RANK'], os.environ['LOCAL_RANK'], os.environ['WORLD_SIZE']
        #   - torch.distributed.get_rank(), torch.distributed.get_world_size()
        worker_result = user_func(*args, **kwargs)
        
        # Only rank 0 returns the result
        if rank == 0:
            result_queue.put(('success', worker_result))
        
        dist.destroy_process_group()
        
    except Exception as e:
        if rank == 0:
            import traceback
            traceback.print_exc()
            result_queue.put(('error', e))
        raise

def main():
    import cloudpickle
    
    payload_encoded = sys.argv[1]
    payload = cloudpickle.loads(base64.b64decode(payload_encoded))
    
    func_source = payload.get('func_source')
    func_name = payload.get('func_name')
    func = payload.get('func')
    args = payload['args']
    kwargs = payload['kwargs']
    captured_vars = payload['captured_vars']
    mp_config = payload.get('mp_config')
    env_vars = payload.get('env', {})
    
    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = str(value)
    
    result = {
        'return_value': None,
        'modified_vars': {},
        'exception': None,
    }
    
    exec_globals = {'__builtins__': __builtins__}
    exec_globals.update(captured_vars)
    
    try:
        if func_source:
            exec(func_source, exec_globals)
            user_func = exec_globals[func_name]
        elif func:
            func_globals = dict(func.__globals__)
            func_globals.update(captured_vars)
            user_func = types.FunctionType(
                func.__code__,
                func_globals,
                func.__name__,
                func.__defaults__,
                func.__closure__
            )
            user_func.__kwdefaults__ = func.__kwdefaults__
            exec_globals = func_globals
        else:
            raise RuntimeError("No function source or function object provided")
            
    except Exception as e:
        result['exception'] = RuntimeError(f"Failed to recreate function: {e}")
        result_encoded = base64.b64encode(cloudpickle.dumps(result)).decode()
        print(f"__REMOTE_RESULT__:{result_encoded}")
        return
    
    # Check if multiprocessing is enabled
    if mp_config is not None:
        try:
            import torch
            import torch.multiprocessing as mp
            
            # Determine number of processes
            num_procs = mp_config['num_processes']
            if num_procs == "auto":
                if torch.cuda.is_available():
                    num_procs = torch.cuda.device_count()
                else:
                    num_procs = os.cpu_count() or 1
            
            backend = mp_config['backend']
            master_addr = mp_config['master_addr']
            master_port = mp_config['master_port']
            start_method = mp_config['start_method']
            
            print(f"[pyremote] Starting {num_procs} processes (backend={backend})")
            
            # Set environment variables for distributed
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = str(master_port)
            
            # Filter out non-picklable items from captured_vars for multiprocessing
            mp_captured_vars = {}
            for k, v in captured_vars.items():
                if isinstance(v, types.ModuleType):
                    continue  # Skip modules
                try:
                    import cloudpickle
                    cloudpickle.dumps(v)
                    mp_captured_vars[k] = v
                except Exception:
                    pass  # Skip non-picklable items
            
            # Shared storage for results from rank 0
            mp_context = mp.get_context(start_method)
            result_queue = mp_context.Queue()
            
            # Spawn workers using module-level worker_fn
            processes = []
            
            for rank in range(num_procs):
                p = mp_context.Process(
                    target=worker_fn,
                    args=(rank, num_procs, func_source, func_name, args, kwargs, 
                          mp_captured_vars, result_queue, backend, env_vars)
                )
                p.start()
                processes.append(p)
            
            # Wait for all processes
            for p in processes:
                p.join()
            
            # Get result from rank 0
            if not result_queue.empty():
                status, value = result_queue.get()
                if status == 'success':
                    result['return_value'] = value
                else:
                    result['exception'] = value
            else:
                result['exception'] = RuntimeError("No result returned from rank 0")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            result['exception'] = e
    
    else:
        # Normal single-process execution
        try:
            result['return_value'] = user_func(*args, **kwargs)
            
            for var in captured_vars.keys():
                if var in exec_globals:
                    try:
                        val = exec_globals[var]
                        cloudpickle.dumps(val)
                        result['modified_vars'][var] = val
                    except Exception:
                        pass
                            
        except Exception as e:
            result['exception'] = e
            for var in captured_vars.keys():
                if var in exec_globals:
                    try:
                        val = exec_globals[var]
                        cloudpickle.dumps(val)
                        result['modified_vars'][var] = val
                    except Exception:
                        pass
    
    result_encoded = base64.b64encode(cloudpickle.dumps(result)).decode()
    print(f"__REMOTE_RESULT__:{result_encoded}")

if __name__ == "__main__":
    main()
'''
    
    def _sync_back_vars(self, modified_vars: Dict[str, Any], frame, func: Callable):
        if not modified_vars:
            return
        
        for var, val in modified_vars.items():
            if var in frame.f_locals:
                frame.f_locals[var] = val
        
        ctypes.pythonapi.PyFrame_LocalsToFast(
            ctypes.py_object(frame),
            ctypes.c_int(0)
        )
        
        if func.__closure__ and func.__code__.co_freevars:
            for i, var in enumerate(func.__code__.co_freevars):
                if var in modified_vars:
                    try:
                        func.__closure__[i].cell_contents = modified_vars[var]
                    except (ValueError, AttributeError):
                        pass
    
    def execute(self, func: Callable, args: tuple = (), kwargs: dict = None, 
                caller_frame=None) -> Any:
        kwargs = kwargs or {}
        
        if caller_frame is None:
            caller_frame = inspect.currentframe().f_back.f_back
        
        captured_vars = self._extract_captured_vars(func, caller_frame)
        
        try:
            self._connect()
            self._setup_uv()
            self._setup_venv()
            self._install_dependencies()
            self._run_setup_commands()
            
            result = self._execute_remote(func, args, kwargs, captured_vars)
            
            if result.stderr:
                print(result.stderr, file=sys.stderr, end='')
            
            self._sync_back_vars(result.modified_vars, caller_frame, func)
            
            if result.exception is not None:
                raise result.exception
            
            return result.return_value
            
        finally:
            # Cleanup virtual environment if delete_after_done is True
            if self._client:
                self._cleanup_venv()
            self._disconnect()


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
    install_verbose: bool = False,
    stdout_callback: Optional[Callable[[str], None]] = None,
    multiprocessing: Optional[Union[int, Literal["auto"], MultiprocessingConfig]] = None,
    env: Optional[Dict[str, str]] = None,
) -> Callable:
    """
    Decorator for remote Python function execution over SSH with optional
    multi-GPU/multi-process support.
    
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
    
    Examples:
        # Single GPU execution (default)
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
    """
    ssh_config = SSHConfig(
        host=host,
        username=username,
        port=port,
        password=password,
        key_filename=key_filename,
        key_password=key_password,
        timeout=timeout,
    )
    
    executor = RemoteExecutor(
        ssh_config=ssh_config,
        venv=venv,
        uv=uv,
        dependencies=dependencies,
        python_path=python_path,
        setup_commands=setup_commands,
        install_verbose=install_verbose,
        stdout_callback=stdout_callback,
        multiprocessing=multiprocessing,
        env=env,
    )
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            caller_frame = inspect.currentframe().f_back
            return executor.execute(func, args, kwargs, caller_frame)
        return wrapper
    
    return decorator