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


@dataclass
class VenvConfig:
    """Standard venv configuration."""
    path: str
    create_if_missing: bool = True
    python_base: str = "python3"
    
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
    python_version: Optional[str] = None  # e.g., "3.12", "3.11"
    create_if_missing: bool = True
    install_uv: bool = True  # install uv if not present
    
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
    ):
        self.ssh_config = ssh_config
        self.dependencies = dependencies or []
        self.setup_commands = setup_commands or []
        
        # handle venv config
        if isinstance(venv, str):
            self.venv = VenvConfig(path=venv)
        elif isinstance(venv, VenvConfig):
            self.venv = venv
        else:
            self.venv = None
        
        # handle uv config
        if isinstance(uv, str):
            self.uv = UvConfig(path=uv)
        elif isinstance(uv, UvConfig):
            self.uv = uv
        else:
            self.uv = None
        
        # uv takes precedence over venv
        if self.uv and self.venv:
            self.venv = None
        
        # determine python path
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
    
    def _connect(self):
        """Establish SSH connection."""
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
        """Close SSH connection."""
        if self._client:
            self._client.close()
            self._client = None
    
    def _run_command(self, cmd: str, timeout: int = None) -> tuple:
        """Run a command on remote and return exit_status, stdout, stderr."""
        timeout = timeout or self.ssh_config.timeout
        stdin, stdout, stderr = self._client.exec_command(cmd, timeout=timeout)
        exit_status = stdout.channel.recv_exit_status()
        return exit_status, stdout.read().decode(), stderr.read().decode()
    
    def _setup_uv(self):
        """Setup UV and virtual environment on remote."""
        if not self.uv:
            return
        
        if self._uv_verified:
            return
        
        uv_path = self.uv.path
        
        # expand ~ on remote
        exit_status, expanded_path, _ = self._run_command(f'echo {uv_path}')
        if exit_status == 0:
            uv_path = expanded_path.strip()
            self.uv = UvConfig(
                path=uv_path,
                python_version=self.uv.python_version,
                create_if_missing=self.uv.create_if_missing,
                install_uv=self.uv.install_uv,
            )
            self._python_path = self.uv.python_path
        
        # check if uv is installed
        exit_status, _, _ = self._run_command('command -v uv')
        
        if exit_status != 0:
            if not self.uv.install_uv:
                raise RemoteVenvError(
                    "UV is not installed on remote and install_uv=False"
                )
            
            # install uv
            install_cmd = 'curl -LsSf https://astral.sh/uv/install.sh | sh'
            exit_status, stdout, stderr = self._run_command(install_cmd, timeout=120)
            
            if exit_status != 0:
                raise RemoteVenvError(f"Failed to install UV: {stderr}")
        
        # check if venv exists
        exit_status, _, _ = self._run_command(f'test -f {self.uv.python_path}')
        
        if exit_status != 0:
            if not self.uv.create_if_missing:
                raise RemoteVenvError(
                    f"Virtual environment not found at {uv_path} "
                    f"and create_if_missing=False"
                )
            
            # create venv with uv
            # need to source uv env first
            python_opt = f'--python {self.uv.python_version}' if self.uv.python_version else ''
            create_cmd = f'''
                source $HOME/.local/bin/env 2>/dev/null || true
                uv venv {python_opt} --allow-existing {uv_path}
            '''
            exit_status, stdout, stderr = self._run_command(create_cmd, timeout=180)
            
            if exit_status != 0:
                raise RemoteVenvError(
                    f"Failed to create UV virtual environment at {uv_path}: {stderr}"
                )
            
            # install cloudpickle
            pip_cmd = f'''
                source $HOME/.local/bin/env 2>/dev/null || true
                source {self.uv.activate_path}
                uv pip install cloudpickle
            '''
            exit_status, stdout, stderr = self._run_command(pip_cmd, timeout=120)
            
            if exit_status != 0:
                raise RemoteVenvError(
                    f"Failed to install cloudpickle in UV venv: {stderr}"
                )
        
        self._uv_verified = True
    
    def _setup_venv(self):
        """Setup standard virtual environment on remote."""
        if not self.venv:
            return
        
        if self._venv_verified:
            return
        
        venv_path = self.venv.path
        
        # expand ~ on remote
        exit_status, expanded_path, _ = self._run_command(f'echo {venv_path}')
        if exit_status == 0:
            venv_path = expanded_path.strip()
            self.venv = VenvConfig(
                path=venv_path,
                create_if_missing=self.venv.create_if_missing,
                python_base=self.venv.python_base,
            )
            self._python_path = self.venv.python_path
        
        # check if venv exists
        exit_status, _, _ = self._run_command(f'test -f {self.venv.python_path}')
        
        if exit_status != 0:
            if not self.venv.create_if_missing:
                raise RemoteVenvError(
                    f"Virtual environment not found at {venv_path} "
                    f"and create_if_missing=False"
                )
            
            create_cmd = f'{self.venv.python_base} -m venv {venv_path}'
            exit_status, stdout, stderr = self._run_command(create_cmd, timeout=120)
            
            if exit_status != 0:
                raise RemoteVenvError(
                    f"Failed to create virtual environment at {venv_path}: {stderr}"
                )
            
            pip_cmd = f'{self.venv.pip_path} install --quiet cloudpickle'
            exit_status, stdout, stderr = self._run_command(pip_cmd, timeout=120)
            
            if exit_status != 0:
                raise RemoteVenvError(
                    f"Failed to install cloudpickle in venv: {stderr}"
                )
        
        self._venv_verified = True
    
    def _install_dependencies(self):
        """Install dependencies on remote if needed."""
        if not self.dependencies:
            return
        
        deps_key = ','.join(sorted(self.dependencies)) + str(self.venv) + str(self.uv)
        deps_hash = hashlib.md5(deps_key.encode()).hexdigest()[:8]
        
        if deps_hash in self._deps_installed:
            return
        
        deps_str = ' '.join(f'"{dep}"' for dep in self.dependencies)
        
        if self.uv:
            # use uv pip
            cmd = f'''
                source $HOME/.local/bin/env 2>/dev/null || true
                source {self.uv.activate_path}
                uv pip install {deps_str}
            '''
        elif self.venv:
            cmd = f'{self.venv.pip_path} install --quiet {deps_str}'
        else:
            cmd = f'{self._python_path} -m pip install --quiet {deps_str}'
        
        exit_status, stdout, stderr = self._run_command(cmd, timeout=300)
        
        if exit_status != 0:
            raise RemoteImportError(
                f"Failed to install dependencies {self.dependencies}: {stderr}"
            )
        
        self._deps_installed.add(deps_hash)
    
    def _run_setup_commands(self):
        """Run any setup commands before execution."""
        for cmd in self.setup_commands:
            exit_status, stdout, stderr = self._run_command(cmd)
            if exit_status != 0:
                raise RemoteExecutionError(
                    f"Setup command failed: {cmd}\nstderr: {stderr}"
                )
    
    def _extract_captured_vars(self, func: Callable, frame) -> Dict[str, Any]:
        """Extract all variables that the function references from outer scopes."""
        captured = {}
        code = func.__code__
        
        # get free variables (from closure)
        if code.co_freevars and func.__closure__:
            for i, var in enumerate(code.co_freevars):
                try:
                    val = func.__closure__[i].cell_contents
                    cloudpickle.dumps(val)
                    captured[var] = val
                except (ValueError, Exception):
                    pass
        
        # find variables accessed via LOAD_GLOBAL or STORE_GLOBAL
        global_vars = set()
        for instr in dis.get_instructions(code):
            if instr.opname in ('LOAD_GLOBAL', 'STORE_GLOBAL'):
                global_vars.add(instr.argval)
        
        # capture these from caller's frame
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
    
    def _execute_remote(self, func: Callable, args: tuple, kwargs: dict, 
                        captured_vars: Dict[str, Any]) -> RemoteResult:
        """Execute function on remote via SSH."""
        
        payload = {
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'captured_vars': captured_vars,
        }
        payload_encoded = base64.b64encode(cloudpickle.dumps(payload)).decode()
        
        executor_script = self._build_executor_script()
        script_encoded = base64.b64encode(executor_script.encode()).decode()
        
        # build activation command
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
        
        stdout_content = stdout.read().decode()
        stderr_content = stderr.read().decode()
        
        result_data = None
        output_lines = []
        
        for line in stdout_content.splitlines():
            if line.startswith('__REMOTE_RESULT__:'):
                result_encoded = line.split(':', 1)[1]
                try:
                    result_data = cloudpickle.loads(base64.b64decode(result_encoded))
                except Exception:
                    pass
            else:
                output_lines.append(line)
        
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
        """Build the script that will run on the remote."""
        return '''
import sys
import base64
import traceback

def main():
    import cloudpickle
    import types
    import dis
    
    payload_encoded = sys.argv[1]
    payload = cloudpickle.loads(base64.b64decode(payload_encoded))
    
    func = payload['func']
    args = payload['args']
    kwargs = payload['kwargs']
    captured_vars = payload['captured_vars']
    
    result = {
        'return_value': None,
        'modified_vars': {},
        'exception': None,
    }
    
    # inject captured variables into function's globals
    func_globals = dict(func.__globals__)
    func_globals.update(captured_vars)
    
    # create new function with updated globals
    new_func = types.FunctionType(
        func.__code__,
        func_globals,
        func.__name__,
        func.__defaults__,
        func.__closure__
    )
    new_func.__kwdefaults__ = func.__kwdefaults__
    
    try:
        result['return_value'] = new_func(*args, **kwargs)
        
        # capture all variables that might have been modified
        for var in captured_vars.keys():
            if var in func_globals:
                try:
                    val = func_globals[var]
                    cloudpickle.dumps(val)
                    result['modified_vars'][var] = val
                except Exception:
                    pass
        
        # also capture any global declarations in the function
        for instr in dis.get_instructions(func.__code__):
            if instr.opname == 'STORE_GLOBAL':
                var = instr.argval
                if var in func_globals and var not in result['modified_vars']:
                    try:
                        val = func_globals[var]
                        cloudpickle.dumps(val)
                        result['modified_vars'][var] = val
                    except Exception:
                        pass
                        
    except Exception as e:
        result['exception'] = e
        # still try to capture modified vars on exception
        for var in captured_vars.keys():
            if var in func_globals:
                try:
                    val = func_globals[var]
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
        """Sync modified variables back to the calling frame."""
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
        """Execute a function remotely and return its result."""
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
            
            if result.stdout:
                print(result.stdout, end='')
            if result.stderr:
                print(result.stderr, file=sys.stderr, end='')
            
            self._sync_back_vars(result.modified_vars, caller_frame, func)
            
            if result.exception is not None:
                raise result.exception
            
            return result.return_value
            
        finally:
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
    )
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            caller_frame = inspect.currentframe().f_back
            return executor.execute(func, args, kwargs, caller_frame)
        return wrapper
    
    return decorator