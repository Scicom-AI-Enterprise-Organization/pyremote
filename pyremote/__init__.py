import ast
import inspect
import textwrap
import cloudpickle
import base64
import hashlib
import tempfile
import os
from contextlib import contextmanager
from typing import Any, List, Optional, Dict, Set, Union
from dataclasses import dataclass, field
import paramiko
import sys


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
    """Virtual environment configuration."""
    path: str  # e.g., "~/.venv" or "/opt/myenv"
    create_if_missing: bool = True
    python_base: str = "python3"  # python to use for creating venv
    
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
    locals: Dict[str, Any]
    exception: Optional[Exception] = None


class RemoteContext:
    def __init__(
        self,
        ssh_config: SSHConfig,
        venv: Optional[Union[str, VenvConfig]] = None,
        dependencies: List[str] = None,
        python_path: Optional[str] = None,
        sync_back: bool = True,
        sync_filter: Optional[callable] = None,
        setup_commands: List[str] = None,
    ):
        self.ssh_config = ssh_config
        self.dependencies = dependencies or []
        self.sync_back = sync_back
        self.sync_filter = sync_filter or (lambda k, v: not k.startswith('_'))
        self.setup_commands = setup_commands or []
        
        # handle venv config
        if isinstance(venv, str):
            self.venv = VenvConfig(path=venv)
        elif isinstance(venv, VenvConfig):
            self.venv = venv
        else:
            self.venv = None
        
        # python_path override takes precedence
        if python_path:
            self._python_path = python_path
        elif self.venv:
            self._python_path = self.venv.python_path
        else:
            self._python_path = "python3"
        
        self._client: Optional[paramiko.SSHClient] = None
        self._local_vars: Dict[str, Any] = {}
        self._frame = None
        self._with_node: Optional[ast.With] = None
        self._source_lines: List[str] = []
        self._deps_installed: Set[str] = set()
        self._venv_verified: bool = False
    
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
    
    def _run_command(self, cmd: str, timeout: int = None) -> tuple[int, str, str]:
        """Run a command on remote and return exit_status, stdout, stderr."""
        timeout = timeout or self.ssh_config.timeout
        stdin, stdout, stderr = self._client.exec_command(cmd, timeout=timeout)
        exit_status = stdout.channel.recv_exit_status()
        return exit_status, stdout.read().decode(), stderr.read().decode()
    
    def _setup_venv(self):
        """Setup virtual environment on remote."""
        if not self.venv:
            return
        
        if self._venv_verified:
            return
        
        venv_path = self.venv.path
        
        # expand ~ on remote
        exit_status, expanded_path, _ = self._run_command(f'echo {venv_path}')
        if exit_status == 0:
            venv_path = expanded_path.strip()
            # update paths with expanded version
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
            
            # create venv
            create_cmd = f'{self.venv.python_base} -m venv {venv_path}'
            exit_status, stdout, stderr = self._run_command(create_cmd, timeout=120)
            
            if exit_status != 0:
                raise RemoteVenvError(
                    f"Failed to create virtual environment at {venv_path}: {stderr}"
                )
            
            # install cloudpickle in the new venv (required for execution)
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
        
        # create a hash of dependencies to check if already installed
        deps_key = ','.join(sorted(self.dependencies)) + str(self.venv)
        deps_hash = hashlib.md5(deps_key.encode()).hexdigest()[:8]
        
        if deps_hash in self._deps_installed:
            return
        
        # use venv pip or system pip
        if self.venv:
            pip_cmd = self.venv.pip_path
        else:
            pip_cmd = f'{self._python_path} -m pip'
        
        deps_str = ' '.join(f'"{dep}"' for dep in self.dependencies)
        cmd = f'{pip_cmd} install --quiet {deps_str}'
        
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
    
    def _capture_context(self):
        """Capture local variables from calling frame."""
        frame = inspect.currentframe().f_back.f_back
        self._frame = frame
        
        for k, v in frame.f_locals.items():
            if self.sync_filter(k, v) and self._is_serializable(v):
                self._local_vars[k] = v
        
        for k, v in frame.f_globals.items():
            if self.sync_filter(k, v) and self._is_serializable(v):
                if k not in self._local_vars:
                    self._local_vars[k] = v
    
    def _extract_with_body(self) -> str:
        """Extract the code inside the with block using AST."""
        frame = self._frame
        
        try:
            filename = frame.f_code.co_filename
            
            if filename == '<stdin>' or filename.startswith('<'):
                raise RemoteExecutionError(
                    "Cannot extract source from interactive mode. "
                    "Please run from a .py file."
                )
            
            with open(filename, 'r') as f:
                source = f.read()
            
            self._source_lines = source.splitlines()
            tree = ast.parse(source, filename=filename)
            
        except OSError as e:
            raise RemoteExecutionError(f"Cannot read source file: {e}") from e
        
        current_line = frame.f_lineno
        
        with_nodes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                with_nodes.append(node)
        
        target_node = None
        for node in with_nodes:
            if node.lineno <= current_line:
                if node.body:
                    body_end = max(
                        getattr(n, 'end_lineno', n.lineno) 
                        for n in ast.walk(node) 
                        if hasattr(n, 'lineno')
                    )
                    if current_line <= body_end:
                        if target_node is None or node.lineno > target_node.lineno:
                            target_node = node
        
        if target_node is None:
            raise RemoteExecutionError("Could not find enclosing with block")
        
        self._with_node = target_node
        
        body_lines = []
        for stmt in target_node.body:
            body_lines.append(ast.unparse(stmt))
        
        return '\n'.join(body_lines)
    
    def _execute_remote(self, code: str) -> RemoteResult:
        """Execute code on remote via SSH."""
        
        executor_script = self._build_executor_script(code)
        script_encoded = base64.b64encode(executor_script.encode()).decode()
        locals_encoded = base64.b64encode(cloudpickle.dumps(self._local_vars)).decode()
        
        # build command with proper venv activation
        if self.venv:
            # use bash -c with source to properly activate venv
            python_cmd = f'source {self.venv.activate_path} && python3'
        else:
            python_cmd = self._python_path
        
        remote_cmd = f'''bash -c '
{python_cmd} -c "
import base64, sys
script = base64.b64decode(\\"{script_encoded}\\").decode()
exec(script)
" "{locals_encoded}"
'
'''
        
        stdin, stdout, stderr = self._client.exec_command(
            remote_cmd, 
            timeout=self.ssh_config.timeout * 10,
            get_pty=True,
        )
        
        stdout_content = stdout.read().decode()
        stderr_content = stderr.read().decode()
        exit_status = stdout.channel.recv_exit_status()
        
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
        
        if result_data is None:
            return RemoteResult(
                stdout=actual_stdout,
                stderr=stderr_content,
                locals={},
                exception=RemoteExecutionError(
                    f"Failed to get result from remote.\n"
                    f"stdout: {actual_stdout}\n"
                    f"stderr: {stderr_content}"
                )
            )
        
        return RemoteResult(
            stdout=actual_stdout,
            stderr=stderr_content,
            locals=result_data.get('locals', {}),
            exception=result_data.get('exception'),
        )
    
    def _build_executor_script(self, code: str) -> str:
        """Build the script that will run on the remote."""
        # escape the code for embedding
        escaped_code = code.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
        
        return f'''
import sys
import base64
import traceback

def main():
    import cloudpickle
    
    locals_encoded = sys.argv[1]
    local_vars = cloudpickle.loads(base64.b64decode(locals_encoded))
    
    result = {{
        'locals': {{}},
        'exception': None,
    }}
    
    exec_globals = {{'__builtins__': __builtins__, '__name__': '__remote__'}}
    exec_locals = dict(local_vars)
    
    try:
        exec("""{escaped_code}""", exec_globals, exec_locals)
        
        for k, v in exec_locals.items():
            if not k.startswith('_'):
                try:
                    cloudpickle.dumps(v)
                    result['locals'][k] = v
                except:
                    pass
                    
    except Exception as e:
        result['exception'] = e
        for k, v in exec_locals.items():
            if not k.startswith('_'):
                try:
                    cloudpickle.dumps(v)
                    result['locals'][k] = v
                except:
                    pass
    
    result_encoded = base64.b64encode(cloudpickle.dumps(result)).decode()
    print(f"__REMOTE_RESULT__:{{result_encoded}}")

if __name__ == "__main__":
    main()
'''
    
    @staticmethod
    def _is_serializable(obj: Any) -> bool:
        """Check if object can be serialized with cloudpickle."""
        try:
            cloudpickle.dumps(obj)
            return True
        except Exception:
            return False
    
    def __enter__(self):
        self._connect()
        self._setup_venv()
        self._install_dependencies()
        self._run_setup_commands()
        self._capture_context()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                return False
            
            code = self._extract_with_body()
            result = self._execute_remote(code)
            
            if result.stdout:
                print(result.stdout, end='')
            if result.stderr:
                print(result.stderr, file=sys.stderr, end='')
            
            if self.sync_back and result.locals:
                import ctypes
                self._frame.f_locals.update(result.locals)
                ctypes.pythonapi.PyFrame_LocalsToFast(
                    ctypes.py_object(self._frame), 
                    ctypes.c_int(0)
                )
            
            if result.exception is not None:
                raise result.exception
            
        finally:
            self._disconnect()
        
        return False


def remote(
    host: str,
    username: str,
    port: int = 22,
    password: Optional[str] = None,
    key_filename: Optional[str] = None,
    key_password: Optional[str] = None,
    timeout: int = 30,
    venv: Optional[Union[str, VenvConfig]] = None,
    dependencies: List[str] = None,
    python_path: Optional[str] = None,
    sync_back: bool = True,
    setup_commands: List[str] = None,
) -> RemoteContext:
    """
    Context manager for remote Python code execution over SSH.
    
    Args:
        host: Remote hostname or IP
        username: SSH username
        port: SSH port (default 22)
        password: SSH password (optional if using key)
        key_filename: Path to SSH private key (optional)
        key_password: Passphrase for SSH key (optional)
        timeout: Connection timeout in seconds
        venv: Virtual environment path (str) or VenvConfig object
              e.g., "~/.venv" or VenvConfig(path="/opt/myenv", create_if_missing=False)
        dependencies: List of pip packages to install on remote
        python_path: Override python interpreter path (takes precedence over venv)
        sync_back: Whether to sync variables back to local scope
        setup_commands: List of shell commands to run before execution
    
    Examples:
        # Using existing venv
        with remote("server.com", "user", password="pass", venv="~/.venv"):
            import numpy as np
            result = np.array([1, 2, 3])
        
        # Auto-create venv if missing
        with remote("server.com", "user", password="pass", 
                    venv=VenvConfig(path="~/myproject/.venv", create_if_missing=True),
                    dependencies=["numpy", "pandas"]):
            import pandas as pd
            df = pd.DataFrame({'a': [1, 2, 3]})
        
        # Use specific python path directly
        with remote("server.com", "user", password="pass",
                    python_path="/opt/conda/envs/ml/bin/python"):
            import torch
            x = torch.tensor([1, 2, 3])
        
        # With setup commands (e.g., conda activate)
        with remote("server.com", "user", password="pass",
                    setup_commands=["source /opt/conda/etc/profile.d/conda.sh"],
                    python_path="/opt/conda/envs/ml/bin/python"):
            import tensorflow as tf
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
    
    return RemoteContext(
        ssh_config=ssh_config,
        venv=venv,
        dependencies=dependencies,
        python_path=python_path,
        sync_back=sync_back,
        setup_commands=setup_commands,
    )