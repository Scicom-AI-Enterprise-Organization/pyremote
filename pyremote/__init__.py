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


# Global executors for Jupyter notebook cell magic (one per profile)
_global_executors: Dict[str, 'RemoteExecutor'] = {}
# Default profile (the first one registered)
_default_profile: Optional[str] = None
# Global namespace for remote cell variables (persists across cells and profiles)
_remote_cell_namespace: Dict[str, Any] = {}


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
        self._persistent_shell = None  # paramiko Channel for persistent Python REPL
    
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
        if self._persistent_shell:
            try:
                self._persistent_shell.close()
            except Exception:
                pass
            self._persistent_shell = None
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
                        captured_vars: Dict[str, Any], func_source: Optional[str] = None) -> RemoteResult:

        if func_source is None:
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
            func_result = user_func(*args, **kwargs)

            # Handle special case for remotecell which returns (locals_dict, result_value)
            if isinstance(func_result, tuple) and len(func_result) == 2:
                locals_dict, actual_result = func_result
                if isinstance(locals_dict, dict):
                    # This is from remotecell - merge locals into modified_vars
                    # Include modules so imports persist across cells
                    for var, val in locals_dict.items():
                        try:
                            cloudpickle.dumps(val)
                            result['modified_vars'][var] = val
                        except Exception:
                            pass
                    result['return_value'] = actual_result
                else:
                    result['return_value'] = func_result
            else:
                result['return_value'] = func_result

            # Also capture any modified variables from exec_globals
            import types
            for var, val in exec_globals.items():
                if var.startswith('__') and var.endswith('__'):
                    continue  # Skip builtins
                if var == func_name:
                    continue  # Skip the function itself
                if isinstance(val, types.ModuleType):
                    continue  # Skip modules
                if var in result['modified_vars']:
                    continue  # Already captured from locals
                try:
                    cloudpickle.dumps(val)
                    result['modified_vars'][var] = val
                except Exception:
                    pass

        except Exception as e:
            result['exception'] = e
    
    result_encoded = base64.b64encode(cloudpickle.dumps(result)).decode()
    print(f"__REMOTE_RESULT__:{result_encoded}")

if __name__ == "__main__":
    main()
'''
    
    def _start_persistent_session(self):
        """Start a persistent Python REPL on the remote via invoke_shell().

        This creates an interactive shell, activates the venv, and starts
        a Python session with a namespace dict for variable persistence.
        """
        import time as _time

        shell = self._client.invoke_shell(term='dumb')
        shell.settimeout(self.ssh_config.timeout * 10)

        # Wait for shell to be ready
        _time.sleep(0.5)
        while shell.recv_ready():
            shell.recv(4096)

        # Activate venv if configured
        if self.uv:
            activate_cmd = f'source $HOME/.local/bin/env 2>/dev/null || true && source {self.uv.activate_path}'
            python_cmd = 'python3'
        elif self.venv:
            activate_cmd = f'source {self.venv.activate_path}'
            python_cmd = 'python3'
        else:
            activate_cmd = ''
            python_cmd = self._python_path

        if activate_cmd:
            shell.sendall(f'{activate_cmd}\n'.encode())
            _time.sleep(0.5)
            while shell.recv_ready():
                shell.recv(4096)

        # Set environment variables
        for key, value in self.env.items():
            shell.sendall(f'export {key}="{value}"\n'.encode())
        if self.env:
            _time.sleep(0.3)
            while shell.recv_ready():
                shell.recv(4096)

        # Start Python REPL with -u for unbuffered output
        shell.sendall(f'{python_cmd} -u -i\n'.encode())
        _time.sleep(1.0)
        while shell.recv_ready():
            shell.recv(4096)

        # Initialize the persistent namespace and imports
        init_code = (
            'import base64, sys\n'
            '__remote_namespace__ = {"__builtins__": __builtins__}\n'
            'print("__PYREMOTE_READY__")\n'
        )
        shell.sendall(init_code.encode())

        # Wait for ready marker
        buffer = ''
        start = _time.time()
        while _time.time() - start < self.ssh_config.timeout:
            if shell.recv_ready():
                chunk = shell.recv(4096).decode('utf-8', errors='replace')
                buffer += chunk
                if '__PYREMOTE_READY__' in buffer:
                    break
            else:
                _time.sleep(0.05)
        else:
            raise RemoteConnectionError(
                f"Persistent session failed to start. Output:\n{buffer}"
            )

        self._persistent_shell = shell

    def _execute_in_persistent_session(self, cell_code: str) -> Any:
        """Execute cell code in the persistent Python REPL session.

        Variables persist in __remote_namespace__ across calls.
        Returns the display value (last expression) or None.
        """
        import time as _time
        import uuid

        shell = self._persistent_shell
        marker = f'__PYREMOTE_EXEC_END_{uuid.uuid4().hex[:12]}__'

        # Wrap code in try/except with marker, handle last expression for display
        lines = cell_code.strip().split('\n')
        last_line = lines[-1].strip() if lines else ''

        is_expression = (
            last_line and
            not last_line.startswith(('def ', 'class ', 'import ', 'from ',
                                     'if ', 'for ', 'while ', 'with ',
                                     'try:', 'except', 'finally:', '@',
                                     'print(', 'print (')) and
            not last_line.endswith(':') and
            '=' not in last_line.split('#')[0]
        )

        if is_expression:
            # Replace last line with assignment + print
            exec_code = '\n'.join(lines[:-1] + [
                f'__result_val__ = {last_line}',
                'if __result_val__ is not None: print(repr(__result_val__))',
            ])
        else:
            exec_code = cell_code

        # Wrap in try/except
        wrapped = (
            'try:\n'
            + '\n'.join('    ' + l for l in exec_code.split('\n'))
            + '\nexcept Exception:\n'
            '    import traceback\n'
            '    traceback.print_exc()\n'
            f'print("{marker}")\n'
        )

        # Base64 encode to avoid escaping issues
        code_encoded = base64.b64encode(wrapped.encode()).decode()
        cmd = (
            f"__user_code__ = base64.b64decode('{code_encoded}').decode()\n"
            f"exec(__user_code__, __remote_namespace__)\n"
        )

        # Noise patterns to filter: echoed commands, REPL prompts, our internals
        _noise_fragments = (
            '__user_code__', 'exec(__user_code__', 'base64.b64decode(',
            '__remote_namespace__', "').decode()",
        )

        def _is_noise(s):
            """Check if a line is shell echo noise rather than user output."""
            s = s.strip()
            if not s:
                return True
            # Pure REPL prompts
            if s in ('>>>', '...'):
                return True
            # Strip leading prompt and check remainder
            for prefix in ('>>> ', '... '):
                if s.startswith(prefix):
                    s = s[len(prefix):]
                    break
            if not s.strip():
                return True
            for frag in _noise_fragments:
                if frag in s:
                    return True
            return False

        # Clear any pending output
        while shell.recv_ready():
            shell.recv(4096)

        shell.sendall(cmd.encode())

        # Read output until marker
        buffer = ''
        output_lines = []
        start = _time.time()
        timeout = self.ssh_config.timeout * 10

        while _time.time() - start < timeout:
            if shell.recv_ready():
                chunk = shell.recv(4096).decode('utf-8', errors='replace')
                buffer += chunk

                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)

                    if marker in line:
                        self._persistent_shell = shell
                        return None

                    if _is_noise(line):
                        continue

                    # Strip leading REPL prompt if present
                    display = line
                    for prefix in ('>>> ', '... '):
                        if display.lstrip().startswith(prefix):
                            display = display.lstrip()[len(prefix):]
                            break

                    output_lines.append(display)
                    print(display, flush=True)
                    if self.stdout_callback:
                        self.stdout_callback(display)

                # Check if marker is in buffer (no trailing newline)
                if marker in buffer:
                    pre_marker = buffer.split(marker)[0]
                    for pline in pre_marker.split('\n'):
                        if not _is_noise(pline) and pline.strip():
                            print(pline.strip(), flush=True)
                            if self.stdout_callback:
                                self.stdout_callback(pline.strip())
                    self._persistent_shell = shell
                    return None
            else:
                _time.sleep(0.05)

        raise RemoteExecutionError(
            f"Persistent session execution timed out after {timeout}s.\n"
            f"Partial output:\n" + '\n'.join(output_lines)
        )

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
                caller_frame=None, func_source: Optional[str] = None,
                extra_captured_vars: Optional[Dict[str, Any]] = None,
                return_full_result: bool = False) -> Any:
        kwargs = kwargs or {}

        if caller_frame is None:
            caller_frame = inspect.currentframe().f_back.f_back

        sync_vars = getattr(self, 'sync_variables', True)

        if sync_vars:
            captured_vars = self._extract_captured_vars(func, caller_frame)
        else:
            captured_vars = {}

        # Merge in extra captured vars (for persistent namespace in Jupyter)
        if extra_captured_vars and sync_vars:
            captured_vars.update(extra_captured_vars)

        # Track if we connected in this call (vs. reusing existing connection)
        was_connected = self._client is not None

        try:
            if not was_connected:
                self._connect()
                self._setup_uv()
                self._setup_venv()
                self._install_dependencies()
                self._run_setup_commands()

            result = self._execute_remote(func, args, kwargs, captured_vars, func_source=func_source)

            if result.stderr:
                print(result.stderr, file=sys.stderr, end='')

            if sync_vars:
                self._sync_back_vars(result.modified_vars, caller_frame, func)

            if result.exception is not None:
                raise result.exception

            # Return full result for Jupyter mode (to get modified_vars)
            # Otherwise just return the value (for backward compatibility)
            return result if return_full_result else result.return_value

        finally:
            # Only cleanup and disconnect if we connected in this call
            # (keep connection alive for jupyter_mode)
            if not was_connected:
                if self._client:
                    self._cleanup_venv()
                self._disconnect()

    def close(self):
        """Close the remote connection and cleanup resources.

        This is useful when using jupyter_mode to properly close the persistent
        connection when you're done.

        Example:
            remote(..., jupyter_mode=True)

            # ... use %%remotecell ...

            # When done:
            from pyremote import close_remote
            close_remote()
        """
        if self._client:
            self._cleanup_venv()
            self._disconnect()
            print("Remote connection closed.")


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
    sync_variables: bool = True,
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
        sync_variables: If True (default), sync local variables to remote and
            sync modified variables back. Set to False to prevent variable
            syncing, which avoids pickling large objects (e.g. CUDA tensors)
            that can cause timeouts.

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
    global _global_executors, _default_profile

    # If called without host/username, raise error
    if host is None or username is None:
        raise ValueError("host and username are required")

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

    # Store sync_variables setting on executor for remotecell magic
    executor.sync_variables = sync_variables

    # Set global executor for notebook magic
    _global_executors[jupyter_profile] = executor

    # Set default profile if this is the first one
    if _default_profile is None:
        _default_profile = jupyter_profile

    # If jupyter_mode is enabled, set up environment immediately and register magic
    if jupyter_mode:
        try:
            executor._connect()
            executor._setup_uv()
            executor._setup_venv()
            executor._install_dependencies()
            executor._run_setup_commands()

            # Register the cell magic
            remotecell()

            # Don't disconnect - keep the connection for cell executions
            # The connection will be reused in _execute_remote

        except Exception as e:
            # Clean up on error
            executor._disconnect()
            raise RuntimeError(f"Failed to set up Jupyter mode: {e}") from e

        return None  # No decorator needed in Jupyter mode

    # If _return_executor is True, just return the executor (for notebook magic setup)
    if _return_executor:
        return executor

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            caller_frame = inspect.currentframe().f_back
            return executor.execute(func, args, kwargs, caller_frame)
        return wrapper

    return decorator


class RemoteCellMagic:
    """IPython cell magic for executing notebook cells remotely."""

    def __init__(self):
        try:
            from IPython.core.magic import Magics, cell_magic, magics_class
            self._magics_available = True
        except ImportError:
            self._magics_available = False

    def register(self):
        """Register the %%remotecell magic with IPython."""
        if not self._magics_available:
            raise ImportError(
                "IPython is required for cell magic support. "
                "Install it with: pip install ipython"
            )

        from IPython.core.magic import Magics, cell_magic, magics_class
        from IPython import get_ipython

        @magics_class
        class RemoteCellMagics(Magics):
            @cell_magic
            def remotecell(self, line, cell):
                """Execute cell code on remote server.

                Usage:
                    %%remotecell
                    import numpy as np
                    result = np.array([1, 2, 3])
                    result  # This will be returned and displayed

                    %%remotecell --profile gpu2
                    # Execute on a specific profile
                """
                global _global_executors, _default_profile

                # Parse line arguments for --profile
                profile_name = _default_profile
                if line and line.strip():
                    parts = line.strip().split()
                    for i, part in enumerate(parts):
                        if part == '--profile' and i + 1 < len(parts):
                            profile_name = parts[i + 1]
                            break

                if not _global_executors:
                    raise RuntimeError(
                        "No remote executor configured. "
                        "Call remote(host=..., username=..., jupyter_mode=True) first to configure the connection."
                    )

                if profile_name not in _global_executors:
                    available = ', '.join(_global_executors.keys())
                    raise RuntimeError(
                        f"Profile '{profile_name}' not found. Available profiles: {available}"
                    )

                executor = _global_executors[profile_name]
                sync_vars = getattr(executor, 'sync_variables', True)

                # When sync_variables=False, use persistent session
                # Variables persist in the remote Python REPL's namespace
                if not sync_vars:
                    try:
                        if executor._persistent_shell is None:
                            executor._start_persistent_session()
                        return executor._execute_in_persistent_session(cell)
                    except Exception as e:
                        print(f"Error executing remote cell: {e}", file=sys.stderr)
                        raise

                # --- sync_variables=True path: pickle-based execution ---

                # Get the IPython user namespace to pass as globals
                ip = get_ipython()
                user_ns = ip.user_ns if ip else {}

                # Create a function from the cell code
                # We need to capture the last expression as return value
                cell_lines = cell.strip().split('\n')

                # Try to detect if the last line is an expression
                last_line = cell_lines[-1].strip() if cell_lines else ""
                is_expression = (
                    last_line and
                    not last_line.startswith(('def ', 'class ', 'import ', 'from ',
                                             'if ', 'for ', 'while ', 'with ',
                                             'try:', 'except', 'finally:', '@')) and
                    not last_line.endswith(':') and
                    '=' not in last_line.split('#')[0]  # Not an assignment
                )

                if is_expression:
                    # Modify to return the last expression
                    cell_code = '\n'.join(cell_lines[:-1] + [f'__result__ = {last_line}'])
                    cell_code += '\n__result__'
                else:
                    cell_code = cell

                # Wrap cell code in a function that returns local variables
                # This ensures variables defined in the cell are captured
                func_code = f"""
def __remotecell_func__():
{textwrap.indent(cell_code, '    ')}
    # Return both the result and all local variables
    __locals_dict__ = {{k: v for k, v in locals().items() if not k.startswith('_')}}
    return (__locals_dict__, locals().get('__result__'))
"""

                # Compile and execute to get the function
                exec_globals = {'__builtins__': __builtins__}
                exec_globals.update(user_ns)
                exec(func_code, exec_globals)
                cell_func = exec_globals['__remotecell_func__']

                # Get caller frame for variable capture
                frame = inspect.currentframe()
                if frame and frame.f_back:
                    caller_frame = frame.f_back
                else:
                    caller_frame = None

                # Execute remotely, passing the source code directly
                # (inspect.getsource won't work for exec'd functions)
                # Also pass the persistent namespace for cross-cell variables
                global _remote_cell_namespace

                # Merge local IPython namespace with remote namespace
                # This allows local variables to be accessible remotely
                import types
                combined_namespace = {}
                if ip:
                    for var, val in ip.user_ns.items():
                        # Skip IPython internals, modules, and magic variables
                        if var.startswith('_') or isinstance(val, types.ModuleType):
                            continue
                        # Skip IPython builtins and functions
                        if var in ('In', 'Out', 'get_ipython', 'exit', 'quit'):
                            continue
                        # Skip functions, classes, and other code objects
                        if isinstance(val, (types.FunctionType, types.MethodType, type)):
                            continue
                        # Skip callables (methods, lambdas, etc)
                        if callable(val) and not hasattr(val, '__array__'):  # Allow numpy arrays
                            continue
                        try:
                            import cloudpickle
                            cloudpickle.dumps(val)
                            combined_namespace[var] = val
                        except Exception:
                            pass

                # Remote namespace takes precedence over local
                combined_namespace.update(_remote_cell_namespace)

                try:
                    result = executor.execute(
                        cell_func,
                        args=(),
                        kwargs={},
                        caller_frame=caller_frame,
                        func_source=func_code,
                        extra_captured_vars=combined_namespace,
                        return_full_result=True  # Get full result with modified_vars
                    )

                    # Update persistent namespace with modified variables
                    # This allows variables defined in remotecell to persist across cells
                    if result.modified_vars:
                        _remote_cell_namespace.update(result.modified_vars)

                    # Also update IPython namespace if available
                    # This syncs remote variables back to local
                    if ip and result.modified_vars:
                        for var, val in result.modified_vars.items():
                            if not var.startswith('_'):  # Don't expose private vars
                                ip.user_ns[var] = val

                    # Return the actual result value
                    return result.return_value

                except Exception as e:
                    print(f"Error executing remote cell: {e}", file=sys.stderr)
                    raise

        # Get IPython instance and register the magic
        ip = get_ipython()
        if ip is None:
            raise RuntimeError("Not running in an IPython environment")

        ip.register_magics(RemoteCellMagics)


# Create a singleton instance
_remote_cell_magic = RemoteCellMagic()


def remotecell():
    """Register the %%remotecell magic for Jupyter notebooks.

    This function should be called after configuring the remote executor
    with the remote() function.

    Example:
        from pyremote import remote, remotecell

        # Configure remote connection
        remote(
            host="server",
            username="user",
            password="xxx",
            dependencies=["torch", "numpy"]
        )

        # Register the magic
        remotecell()

        # Now use %%remotecell in notebook cells
        %%remotecell
        import torch
        print(torch.cuda.is_available())
    """
    _remote_cell_magic.register()


def close_remote(profile: Optional[str] = None):
    """Close remote connection(s) used by %%remotecell magic.

    This is useful when using jupyter_mode=True to properly close the persistent
    connection when you're done.

    Args:
        profile: Optional profile name to close. If None, closes all profiles.

    Examples:
        from pyremote import remote, close_remote

        # Configure profiles
        remote(host="server1", username="user", password="xxx",
               jupyter_mode=True, jupyter_profile="gpu1")
        remote(host="server2", username="user", password="xxx",
               jupyter_mode=True, jupyter_profile="gpu2")

        # Close specific profile
        close_remote(profile="gpu1")

        # Close all profiles
        close_remote()
    """
    global _global_executors, _default_profile, _remote_cell_namespace

    if not _global_executors:
        print("No remote connections to close.")
        return

    if profile is not None:
        # Close specific profile
        if profile in _global_executors:
            _global_executors[profile].close()
            del _global_executors[profile]
            if _default_profile == profile:
                _default_profile = next(iter(_global_executors.keys())) if _global_executors else None
        else:
            print(f"Profile '{profile}' not found.")
    else:
        # Close all profiles
        for prof, executor in _global_executors.items():
            executor.close()
        _global_executors = {}
        _default_profile = None
        _remote_cell_namespace = {}  # Clear namespace when closing all


# Exports
__all__ = [
    'remote',
    'remotecell',
    'close_remote',
    'RemoteExecutor',
    'RemoteResult',
    'SSHConfig',
    'VenvConfig',
    'UvConfig',
    'MultiprocessingConfig',
    'RemoteExecutionError',
    'RemoteImportError',
    'RemoteConnectionError',
    'RemoteVenvError',
]