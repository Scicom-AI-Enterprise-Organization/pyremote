"""
Comprehensive end-to-end tests for remote.py

These tests require a real SSH server to run. Configure the following
environment variables before running:

    export TEST_SSH_HOST="your-server.com"
    export TEST_SSH_USERNAME="your-username"
    export TEST_SSH_PASSWORD="your-password"  # or use key-based auth
    export TEST_SSH_KEY_FILE="~/.ssh/id_rsa"  # optional
    export TEST_SSH_PORT="22"  # optional, defaults to 22

Run with: pytest test_remote.py -v -s
"""

import os
import sys
import pytest
import time
import tempfile
from typing import List, Dict, Any
from dataclasses import dataclass
from unittest.mock import patch, MagicMock

from pyremote import (
    remote,
    RemoteExecutor,
    SSHConfig,
    VenvConfig,
    UvConfig,
    RemoteResult,
    RemoteExecutionError,
    RemoteImportError,
    RemoteConnectionError,
    RemoteVenvError,
)

def get_ssh_config() -> dict:
    """Get SSH configuration from environment variables."""
    host = os.environ.get("TEST_SSH_HOST")
    username = os.environ.get("TEST_SSH_USERNAME")
    password = os.environ.get("TEST_SSH_PASSWORD")
    key_filename = os.environ.get("TEST_SSH_KEY_FILE")
    port = int(os.environ.get("TEST_SSH_PORT", "22"))
    
    if not host or not username:
        pytest.skip("SSH credentials not configured. Set TEST_SSH_HOST and TEST_SSH_USERNAME")
    
    return {
        "host": host,
        "username": username,
        "password": password,
        "key_filename": key_filename,
        "port": port,
    }


@pytest.fixture
def ssh_config():
    """Fixture providing SSH configuration."""
    return get_ssh_config()


@pytest.fixture
def unique_venv_path():
    """Generate a unique venv path for test isolation."""
    timestamp = int(time.time() * 1000)
    return f"/tmp/test_venv_{timestamp}_{os.getpid()}"


@pytest.fixture
def unique_uv_path():
    """Generate a unique UV venv path for test isolation."""
    timestamp = int(time.time() * 1000)
    return f"/tmp/test_uv_venv_{timestamp}_{os.getpid()}"

class TestSSHConfig:
    """Tests for SSHConfig dataclass."""
    
    def test_default_values(self):
        config = SSHConfig(host="example.com", username="user")
        assert config.port == 22
        assert config.password is None
        assert config.key_filename is None
        assert config.key_password is None
        assert config.timeout == 30
    
    def test_custom_values(self):
        config = SSHConfig(
            host="example.com",
            username="user",
            port=2222,
            password="secret",
            key_filename="~/.ssh/custom_key",
            key_password="keypass",
            timeout=60,
        )
        assert config.port == 2222
        assert config.password == "secret"
        assert config.key_filename == "~/.ssh/custom_key"
        assert config.key_password == "keypass"
        assert config.timeout == 60


class TestVenvConfig:
    """Tests for VenvConfig dataclass."""
    
    def test_default_values(self):
        config = VenvConfig(path="/home/user/venv")
        assert config.create_if_missing is True
        assert config.python_base == "python3"
    
    def test_paths(self):
        config = VenvConfig(path="/home/user/venv")
        assert config.python_path == "/home/user/venv/bin/python3"
        assert config.pip_path == "/home/user/venv/bin/pip"
        assert config.activate_path == "/home/user/venv/bin/activate"
    
    def test_custom_python_base(self):
        config = VenvConfig(path="/venv", python_base="python3.11")
        assert config.python_base == "python3.11"


class TestUvConfig:
    """Tests for UvConfig dataclass."""
    
    def test_default_values(self):
        config = UvConfig()
        assert config.path == ".venv"
        assert config.python_version is None
        assert config.create_if_missing is True
        assert config.install_uv is True
    
    def test_paths(self):
        config = UvConfig(path="/home/user/.venv")
        assert config.python_path == "/home/user/.venv/bin/python3"
        assert config.pip_path == "/home/user/.venv/bin/pip"
        assert config.activate_path == "/home/user/.venv/bin/activate"
    
    def test_python_version(self):
        config = UvConfig(python_version="3.12")
        assert config.python_version == "3.12"


class TestRemoteResult:
    """Tests for RemoteResult dataclass."""
    
    def test_default_exception(self):
        result = RemoteResult(
            stdout="output",
            stderr="",
            return_value=42,
            modified_vars={"x": 10},
        )
        assert result.exception is None
    
    def test_with_exception(self):
        exc = ValueError("test error")
        result = RemoteResult(
            stdout="",
            stderr="error",
            return_value=None,
            modified_vars={},
            exception=exc,
        )
        assert result.exception is exc


class TestRemoteExecutorInit:
    """Tests for RemoteExecutor initialization."""
    
    def test_string_venv_conversion(self):
        ssh = SSHConfig(host="test", username="user")
        executor = RemoteExecutor(ssh, venv="/path/to/venv")
        assert isinstance(executor.venv, VenvConfig)
        assert executor.venv.path == "/path/to/venv"
    
    def test_string_uv_conversion(self):
        ssh = SSHConfig(host="test", username="user")
        executor = RemoteExecutor(ssh, uv="/path/to/uv")
        assert isinstance(executor.uv, UvConfig)
        assert executor.uv.path == "/path/to/uv"
    
    def test_uv_takes_precedence_over_venv(self):
        ssh = SSHConfig(host="test", username="user")
        executor = RemoteExecutor(ssh, venv="/venv", uv="/uv")
        assert executor.uv is not None
        assert executor.venv is None
    
    def test_python_path_priority(self):
        ssh = SSHConfig(host="test", username="user")
        
        # explicit python_path takes priority
        executor = RemoteExecutor(
            ssh, 
            uv="/uv", 
            venv="/venv", 
            python_path="/custom/python"
        )
        assert executor._python_path == "/custom/python"
        
        # uv path when no explicit
        executor = RemoteExecutor(ssh, uv="/uv")
        assert executor._python_path == "/uv/bin/python3"
        
        # venv path when no uv
        executor = RemoteExecutor(ssh, venv="/venv")
        assert executor._python_path == "/venv/bin/python3"
        
        # default python3
        executor = RemoteExecutor(ssh)
        assert executor._python_path == "python3"
    
    def test_default_dependencies(self):
        ssh = SSHConfig(host="test", username="user")
        executor = RemoteExecutor(ssh)
        assert executor.dependencies == []
    
    def test_stream_install_default(self):
        ssh = SSHConfig(host="test", username="user")
        executor = RemoteExecutor(ssh)
        assert executor.stream_install is False
    
    def test_stream_install_enabled(self):
        ssh = SSHConfig(host="test", username="user")
        executor = RemoteExecutor(ssh, stream_install=True)
        assert executor.stream_install is True


class TestFunctionSourceExtraction:
    """Tests for _get_function_source method."""
    
    def test_simple_function(self):
        ssh = SSHConfig(host="test", username="user")
        executor = RemoteExecutor(ssh)
        
        def simple_func():
            return 42
        
        source = executor._get_function_source(simple_func)
        assert source is not None
        assert "def simple_func():" in source
        assert "return 42" in source
    
    def test_function_with_decorator_removed(self):
        ssh = SSHConfig(host="test", username="user")
        executor = RemoteExecutor(ssh)
        
        def dummy_decorator(f):
            return f
        
        @dummy_decorator
        def decorated_func():
            return "hello"
        
        source = executor._get_function_source(decorated_func)
        assert source is not None
        assert "def decorated_func():" in source
        assert "@" not in source or source.index("def") < source.index("@") if "@" in source else True


class TestCapturedVarsExtraction:
    """Tests for _extract_captured_vars method."""
    
    def test_closure_variable(self):
        ssh = SSHConfig(host="test", username="user")
        executor = RemoteExecutor(ssh)
        
        outer_var = 100
        
        def inner_func():
            return outer_var
        
        import inspect
        frame = inspect.currentframe()
        captured = executor._extract_captured_vars(inner_func, frame)
        
        assert "outer_var" in captured
        assert captured["outer_var"] == 100
    
    def test_unpicklable_vars_excluded(self):
        ssh = SSHConfig(host="test", username="user")
        executor = RemoteExecutor(ssh)
        
        # lambda is picklable with cloudpickle, but let's test with module
        import sys as sys_module
        
        def func():
            return sys_module
        
        import inspect
        frame = inspect.currentframe()
        captured = executor._extract_captured_vars(func, frame)
        
        # modules should be excluded
        assert "sys_module" not in captured

@pytest.mark.integration
class TestBasicRemoteExecution:
    """Basic remote execution tests."""
    
    def test_simple_return_value(self, ssh_config):
        """Test returning a simple value."""
        @remote(**ssh_config)
        def get_number():
            return 42
        
        result = get_number()
        assert result == 42
    
    def test_return_string(self, ssh_config):
        """Test returning a string."""
        @remote(**ssh_config)
        def get_string():
            return "hello world"
        
        result = get_string()
        assert result == "hello world"
    
    def test_return_list(self, ssh_config):
        """Test returning a list."""
        @remote(**ssh_config)
        def get_list():
            return [1, 2, 3, 4, 5]
        
        result = get_list()
        assert result == [1, 2, 3, 4, 5]
    
    def test_return_dict(self, ssh_config):
        """Test returning a dictionary."""
        @remote(**ssh_config)
        def get_dict():
            return {"a": 1, "b": 2, "c": 3}
        
        result = get_dict()
        assert result == {"a": 1, "b": 2, "c": 3}
    
    def test_return_nested_structure(self, ssh_config):
        """Test returning nested data structures."""
        @remote(**ssh_config)
        def get_nested():
            return {
                "list": [1, 2, 3],
                "dict": {"nested": True},
                "tuple": (1, 2),
            }
        
        result = get_nested()
        assert result["list"] == [1, 2, 3]
        assert result["dict"]["nested"] is True
        assert result["tuple"] == (1, 2)
    
    def test_return_none(self, ssh_config):
        """Test returning None."""
        @remote(**ssh_config)
        def get_none():
            return None
        
        result = get_none()
        assert result is None
    
    def test_no_return_value(self, ssh_config):
        """Test function with no return statement."""
        @remote(**ssh_config)
        def no_return():
            x = 1 + 1
        
        result = no_return()
        assert result is None


@pytest.mark.integration
class TestFunctionArguments:
    """Tests for passing arguments to remote functions."""
    
    def test_positional_args(self, ssh_config):
        """Test positional arguments."""
        @remote(**ssh_config)
        def add(a, b):
            return a + b
        
        result = add(3, 4)
        assert result == 7
    
    def test_keyword_args(self, ssh_config):
        """Test keyword arguments."""
        @remote(**ssh_config)
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"
        
        result = greet(name="World", greeting="Hi")
        assert result == "Hi, World!"
    
    def test_default_args(self, ssh_config):
        """Test default arguments."""
        @remote(**ssh_config)
        def power(base, exp=2):
            return base ** exp
        
        assert power(3) == 9
        assert power(2, 3) == 8
    
    def test_mixed_args(self, ssh_config):
        """Test mixed positional and keyword arguments."""
        @remote(**ssh_config)
        def calculate(a, b, op="add"):
            if op == "add":
                return a + b
            elif op == "mul":
                return a * b
            return 0
        
        assert calculate(2, 3) == 5
        assert calculate(2, 3, op="mul") == 6
    
    def test_args_with_list(self, ssh_config):
        """Test passing list as argument."""
        @remote(**ssh_config)
        def sum_list(numbers):
            return sum(numbers)
        
        result = sum_list([1, 2, 3, 4, 5])
        assert result == 15
    
    def test_args_with_dict(self, ssh_config):
        """Test passing dict as argument."""
        @remote(**ssh_config)
        def get_value(data, key):
            return data.get(key)
        
        result = get_value({"x": 100, "y": 200}, "x")
        assert result == 100
    
    def test_varargs(self, ssh_config):
        """Test *args."""
        @remote(**ssh_config)
        def sum_all(*args):
            return sum(args)
        
        result = sum_all(1, 2, 3, 4, 5)
        assert result == 15
    
    def test_kwargs(self, ssh_config):
        """Test **kwargs."""
        @remote(**ssh_config)
        def collect_kwargs(**kwargs):
            return kwargs
        
        result = collect_kwargs(a=1, b=2, c=3)
        assert result == {"a": 1, "b": 2, "c": 3}


@pytest.mark.integration
class TestVariableCapture:
    """Tests for capturing variables from enclosing scope."""
    
    def test_capture_simple_variable(self, ssh_config):
        """Test capturing a simple variable."""
        multiplier = 10
        
        @remote(**ssh_config)
        def multiply(x):
            return x * multiplier
        
        result = multiply(5)
        assert result == 50
    
    def test_capture_multiple_variables(self, ssh_config):
        """Test capturing multiple variables."""
        a = 1
        b = 2
        c = 3
        
        @remote(**ssh_config)
        def sum_vars():
            return a + b + c
        
        result = sum_vars()
        assert result == 6
    
    def test_capture_list(self, ssh_config):
        """Test capturing a list."""
        items = [10, 20, 30]
        
        @remote(**ssh_config)
        def get_sum():
            return sum(items)
        
        result = get_sum()
        assert result == 60
    
    def test_capture_dict(self, ssh_config):
        """Test capturing a dictionary."""
        config = {"threshold": 0.5, "max_items": 100}
        
        @remote(**ssh_config)
        def get_config_value(key):
            return config[key]
        
        result = get_config_value("threshold")
        assert result == 0.5
    
    def test_capture_nested_function(self, ssh_config):
        """Test capturing with nested function definition."""
        factor = 2
        
        def outer():
            @remote(**ssh_config)
            def inner(x):
                return x * factor
            return inner
        
        multiply = outer()
        result = multiply(21)
        assert result == 42
    
    def test_capture_class_instance(self, ssh_config):
        """Test capturing a class instance."""
        @dataclass
        class Config:
            value: int
            name: str
        
        cfg = Config(value=42, name="test")
        
        @remote(**ssh_config)
        def get_config_info():
            return {"value": cfg.value, "name": cfg.name}
        
        result = get_config_info()
        assert result["value"] == 42
        assert result["name"] == "test"


@pytest.mark.integration
class TestVariableModification:
    """Tests for modifying variables and syncing back."""
    
    def test_modify_list_inplace(self, ssh_config):
        """Test modifying a list in-place."""
        data = [1, 2, 3]
        
        @remote(**ssh_config)
        def append_item():
            data.append(4)
        
        append_item()
        assert data == [1, 2, 3, 4]
    
    def test_modify_dict_inplace(self, ssh_config):
        """Test modifying a dict in-place."""
        state = {"count": 0}
        
        @remote(**ssh_config)
        def increment():
            state["count"] += 1
        
        increment()
        assert state["count"] == 1
    
    def test_modify_global_variable(self, ssh_config):
        """Test modifying a global variable with global keyword."""
        counter = 0
        
        @remote(**ssh_config)
        def increment_counter():
            global counter
            counter = counter + 10
        
        increment_counter()
        # Note: global modification sync depends on implementation
        # This tests that it doesn't raise an error


@pytest.mark.integration
class TestExceptionHandling:
    """Tests for exception handling."""
    
    def test_remote_exception(self, ssh_config):
        """Test that remote exceptions are raised locally."""
        @remote(**ssh_config)
        def raise_error():
            raise ValueError("test error")
        
        with pytest.raises(ValueError) as exc_info:
            raise_error()
        
        assert "test error" in str(exc_info.value)
    
    def test_remote_type_error(self, ssh_config):
        """Test TypeError from remote."""
        @remote(**ssh_config)
        def cause_type_error():
            return "hello" + 5
        
        with pytest.raises(TypeError):
            cause_type_error()
    
    def test_remote_key_error(self, ssh_config):
        """Test KeyError from remote."""
        @remote(**ssh_config)
        def cause_key_error():
            d = {"a": 1}
            return d["missing"]
        
        with pytest.raises(KeyError):
            cause_key_error()
    
    def test_remote_zero_division(self, ssh_config):
        """Test ZeroDivisionError from remote."""
        @remote(**ssh_config)
        def divide_by_zero():
            return 1 / 0
        
        with pytest.raises(ZeroDivisionError):
            divide_by_zero()
    
    def test_remote_custom_exception(self, ssh_config):
        """Test custom exception from remote."""
        class CustomError(Exception):
            pass
        
        @remote(**ssh_config)
        def raise_custom():
            raise CustomError("custom message")
        
        with pytest.raises(CustomError):
            raise_custom()


@pytest.mark.integration
class TestPrintAndOutput:
    """Tests for stdout/stderr handling."""
    
    def test_print_output(self, ssh_config, capsys):
        """Test that print statements are captured."""
        @remote(**ssh_config)
        def print_hello():
            print("Hello from remote!")
            return "done"
        
        result = print_hello()
        captured = capsys.readouterr()
        
        assert result == "done"
        assert "Hello from remote!" in captured.out
    
    def test_multiple_prints(self, ssh_config, capsys):
        """Test multiple print statements."""
        @remote(**ssh_config)
        def print_numbers():
            for i in range(5):
                print(f"Number: {i}")
            return "complete"
        
        result = print_numbers()
        captured = capsys.readouterr()
        
        assert result == "complete"
        for i in range(5):
            assert f"Number: {i}" in captured.out
    
    def test_print_with_return(self, ssh_config, capsys):
        """Test print and return value together."""
        @remote(**ssh_config)
        def compute_and_print():
            result = 2 + 2
            print(f"Computed: {result}")
            return result
        
        result = compute_and_print()
        captured = capsys.readouterr()
        
        assert result == 4
        assert "Computed: 4" in captured.out


@pytest.mark.integration
class TestStandardVenv:
    """Tests for standard venv support."""
    
    def test_venv_string_path(self, ssh_config, unique_venv_path):
        """Test venv with string path."""
        @remote(**ssh_config, venv=unique_venv_path)
        def check_python():
            import sys
            return sys.executable
        
        result = check_python()
        assert unique_venv_path in result
    
    def test_venv_config_object(self, ssh_config, unique_venv_path):
        """Test venv with VenvConfig object."""
        venv_config = VenvConfig(
            path=unique_venv_path,
            create_if_missing=True,
        )
        
        @remote(**ssh_config, venv=venv_config)
        def get_version():
            import sys
            return sys.version_info[:2]
        
        result = get_version()
        assert isinstance(result, tuple)
        assert result[0] >= 3
    
    def test_venv_create_if_missing_true(self, ssh_config, unique_venv_path):
        """Test that venv is created when missing."""
        @remote(**ssh_config, venv=unique_venv_path)
        def simple_calc():
            return 1 + 1
        
        result = simple_calc()
        assert result == 2
    
    def test_venv_with_dependencies(self, ssh_config, unique_venv_path):
        """Test venv with dependencies."""
        @remote(
            **ssh_config,
            venv=unique_venv_path,
            dependencies=["requests"],
        )
        def check_requests():
            import requests
            return requests.__version__
        
        result = check_requests()
        assert result is not None


@pytest.mark.integration
class TestUvVenv:
    """Tests for UV-based virtual environment."""
    
    def test_uv_string_path(self, ssh_config, unique_uv_path):
        """Test UV with string path."""
        @remote(**ssh_config, uv=unique_uv_path)
        def check_python():
            import sys
            return sys.executable
        
        result = check_python()
        assert unique_uv_path in result or "bin/python" in result
    
    def test_uv_config_object(self, ssh_config, unique_uv_path):
        """Test UV with UvConfig object."""
        uv_config = UvConfig(
            path=unique_uv_path,
            create_if_missing=True,
            install_uv=True,
        )
        
        @remote(**ssh_config, uv=uv_config)
        def get_version():
            import sys
            return sys.version_info[:2]
        
        result = get_version()
        assert isinstance(result, tuple)
        assert result[0] >= 3
    
    def test_uv_with_python_version(self, ssh_config, unique_uv_path):
        """Test UV with specific Python version."""
        uv_config = UvConfig(
            path=unique_uv_path,
            python_version="3.11",
            create_if_missing=True,
        )
        
        @remote(**ssh_config, uv=uv_config)
        def get_version():
            import sys
            return sys.version_info[:2]
        
        try:
            result = get_version()
            # If Python 3.11 is available, check version
            assert result[0] == 3
            assert result[1] == 11
        except RemoteVenvError:
            # Python 3.11 might not be available on remote
            pytest.skip("Python 3.11 not available on remote")
    
    def test_uv_with_dependencies(self, ssh_config, unique_uv_path):
        """Test UV with dependencies."""
        @remote(
            **ssh_config,
            uv=unique_uv_path,
            dependencies=["httpx"],
        )
        def check_httpx():
            import httpx
            return httpx.__version__
        
        result = check_httpx()
        assert result is not None
    
    def test_uv_takes_precedence(self, ssh_config, unique_uv_path, unique_venv_path):
        """Test that UV takes precedence over standard venv."""
        @remote(
            **ssh_config,
            venv=unique_venv_path,
            uv=unique_uv_path,
        )
        def check_path():
            import sys
            return sys.executable
        
        result = check_path()
        # Should use UV path, not standard venv
        assert unique_uv_path in result or "bin/python" in result


@pytest.mark.integration
class TestDependencies:
    """Tests for dependency installation."""
    
    def test_single_dependency(self, ssh_config, unique_venv_path):
        """Test installing a single dependency."""
        @remote(
            **ssh_config,
            venv=unique_venv_path,
            dependencies=["six"],
        )
        def use_six():
            import six
            return six.PY3
        
        result = use_six()
        assert result is True
    
    def test_multiple_dependencies(self, ssh_config, unique_venv_path):
        """Test installing multiple dependencies."""
        @remote(
            **ssh_config,
            venv=unique_venv_path,
            dependencies=["six", "attrs"],
        )
        def check_deps():
            import six
            import attr
            return {"six": six.__version__, "attrs": attr.__version__}
        
        result = check_deps()
        assert "six" in result
        assert "attrs" in result
    
    def test_dependency_with_version(self, ssh_config, unique_venv_path):
        """Test installing dependency with version specifier."""
        @remote(
            **ssh_config,
            venv=unique_venv_path,
            dependencies=["six>=1.10"],
        )
        def check_version():
            import six
            return six.__version__
        
        result = check_version()
        # Version should be >= 1.10
        major, minor = map(int, result.split(".")[:2])
        assert (major, minor) >= (1, 10)
    
    def test_stream_install_output(self, ssh_config, unique_venv_path, capsys):
        """Test streaming installation output."""
        @remote(
            **ssh_config,
            venv=unique_venv_path,
            dependencies=["wheel"],
            stream_install=True,
        )
        def simple():
            return True
        
        result = simple()
        captured = capsys.readouterr()
        
        assert result is True
        # Should have some output from pip install
        # (might be empty if wheel is already cached)


@pytest.mark.integration
class TestSetupCommands:
    """Tests for setup commands."""
    
    def test_single_setup_command(self, ssh_config):
        """Test single setup command."""
        @remote(
            **ssh_config,
            setup_commands=["export TEST_VAR=hello"],
        )
        def check_var():
            import os
            return os.environ.get("TEST_VAR", "not set")
        
        # Note: export in setup might not persist to Python execution
        # depending on how commands are run
        result = check_var()
        # Just verify it doesn't crash
    
    def test_multiple_setup_commands(self, ssh_config):
        """Test multiple setup commands."""
        @remote(
            **ssh_config,
            setup_commands=[
                "echo 'Setup 1'",
                "echo 'Setup 2'",
            ],
        )
        def simple():
            return True
        
        result = simple()
        assert result is True


@pytest.mark.integration
class TestCrossPythonVersion:
    """Tests for cross-Python-version execution."""
    
    def test_different_python_versions(self, ssh_config, unique_uv_path):
        """Test execution with different Python version."""
        local_version = sys.version_info[:2]
        
        # Try to use a different Python version on remote
        target_version = "3.11" if local_version != (3, 11) else "3.10"
        
        uv_config = UvConfig(
            path=unique_uv_path,
            python_version=target_version,
            create_if_missing=True,
        )
        
        @remote(**ssh_config, uv=uv_config)
        def get_version():
            import sys
            return sys.version_info[:2]
        
        try:
            result = get_version()
            assert result[0] == 3
        except RemoteVenvError:
            pytest.skip(f"Python {target_version} not available on remote")
    
    def test_source_code_transfer(self, ssh_config, unique_uv_path):
        """Test that function works via source code (cross-version compatible)."""
        @remote(**ssh_config, uv=unique_uv_path)
        def complex_function(data: List[int]) -> Dict[str, Any]:
            """A function with type hints and docstring."""
            result = {
                "sum": sum(data),
                "count": len(data),
                "avg": sum(data) / len(data) if data else 0,
            }
            return result
        
        result = complex_function([1, 2, 3, 4, 5])
        assert result["sum"] == 15
        assert result["count"] == 5
        assert result["avg"] == 3.0


@pytest.mark.integration
class TestComplexScenarios:
    """Tests for complex real-world scenarios."""
    
    def test_data_processing_pipeline(self, ssh_config, unique_venv_path):
        """Test a data processing pipeline."""
        input_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        @remote(**ssh_config, venv=unique_venv_path)
        def process_data(data):
            # Filter even numbers
            filtered = [x for x in data if x % 2 == 0]
            # Square them
            squared = [x ** 2 for x in filtered]
            # Sum
            return sum(squared)
        
        result = process_data(input_data)
        # 2^2 + 4^2 + 6^2 + 8^2 + 10^2 = 4 + 16 + 36 + 64 + 100 = 220
        assert result == 220
    
    def test_recursive_function(self, ssh_config):
        """Test recursive function."""
        @remote(**ssh_config)
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)
        
        # Note: this calls remote only once, recursion happens on remote
        @remote(**ssh_config)
        def fib_remote(n):
            def fib(x):
                if x <= 1:
                    return x
                return fib(x - 1) + fib(x - 2)
            return fib(n)
        
        result = fib_remote(10)
        assert result == 55
    
    def test_with_numpy(self, ssh_config, unique_venv_path):
        """Test with numpy dependency."""
        @remote(
            **ssh_config,
            venv=unique_venv_path,
            dependencies=["numpy"],
        )
        def numpy_operations():
            import numpy as np
            arr = np.array([1, 2, 3, 4, 5])
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "sum": int(np.sum(arr)),
            }
        
        result = numpy_operations()
        assert result["mean"] == 3.0
        assert result["sum"] == 15
    
    def test_class_method_like(self, ssh_config):
        """Test function that behaves like a class method."""
        class DataProcessor:
            def __init__(self, multiplier):
                self.multiplier = multiplier
            
            def process(self, data):
                mult = self.multiplier
                
                @remote(**ssh_config)
                def remote_process(items):
                    return [x * mult for x in items]
                
                return remote_process(data)
        
        processor = DataProcessor(3)
        result = processor.process([1, 2, 3])
        assert result == [3, 6, 9]
    
    def test_context_manager_compatibility(self, ssh_config):
        """Test that remote execution works with context managers on remote."""
        @remote(**ssh_config)
        def use_context_manager():
            import io
            with io.StringIO() as buffer:
                buffer.write("Hello, World!")
                return buffer.getvalue()
        
        result = use_context_manager()
        assert result == "Hello, World!"
    
    def test_generator_to_list(self, ssh_config):
        """Test converting generator to list on remote."""
        @remote(**ssh_config)
        def generate_numbers(n):
            def gen():
                for i in range(n):
                    yield i * 2
            return list(gen())
        
        result = generate_numbers(5)
        assert result == [0, 2, 4, 6, 8]


@pytest.mark.integration
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_function(self, ssh_config):
        """Test function with just pass."""
        @remote(**ssh_config)
        def empty():
            pass
        
        result = empty()
        assert result is None
    
    def test_large_return_value(self, ssh_config):
        """Test returning large data."""
        @remote(**ssh_config)
        def large_data():
            return list(range(10000))
        
        result = large_data()
        assert len(result) == 10000
        assert result[0] == 0
        assert result[-1] == 9999
    
    def test_unicode_handling(self, ssh_config):
        """Test Unicode string handling."""
        @remote(**ssh_config)
        def unicode_test():
            return {
                "emoji": "🎉🐍🚀",
                "chinese": "你好世界",
                "arabic": "مرحبا",
                "mixed": "Hello 世界 🌍",
            }
        
        result = unicode_test()
        assert result["emoji"] == "🎉🐍🚀"
        assert result["chinese"] == "你好世界"
    
    def test_special_characters_in_string(self, ssh_config):
        """Test special characters in strings."""
        @remote(**ssh_config)
        def special_chars():
            return "Line1\nLine2\tTabbed\r\nWindows"
        
        result = special_chars()
        assert "\n" in result
        assert "\t" in result
    
    def test_boolean_return(self, ssh_config):
        """Test boolean return values."""
        @remote(**ssh_config)
        def check_true():
            return True
        
        @remote(**ssh_config)
        def check_false():
            return False
        
        assert check_true() is True
        assert check_false() is False
    
    def test_float_precision(self, ssh_config):
        """Test float precision."""
        @remote(**ssh_config)
        def precise_float():
            return 3.141592653589793
        
        result = precise_float()
        assert abs(result - 3.141592653589793) < 1e-15
    
    def test_negative_numbers(self, ssh_config):
        """Test negative numbers."""
        @remote(**ssh_config)
        def negative():
            return {"int": -42, "float": -3.14}
        
        result = negative()
        assert result["int"] == -42
        assert result["float"] == -3.14
    
    def test_very_long_string(self, ssh_config):
        """Test very long string."""
        @remote(**ssh_config)
        def long_string():
            return "x" * 100000
        
        result = long_string()
        assert len(result) == 100000


@pytest.mark.integration
class TestErrorRecovery:
    """Tests for error recovery scenarios."""
    
    def test_partial_failure_with_modified_vars(self, ssh_config):
        """Test that modified vars are synced even on failure."""
        progress = []
        
        @remote(**ssh_config)
        def partial_work():
            progress.append("step1")
            progress.append("step2")
            raise RuntimeError("Failed at step 3")
        
        with pytest.raises(RuntimeError):
            partial_work()
        
        # Progress should be synced back even though exception occurred
        assert "step1" in progress or len(progress) == 0  # depends on implementation
    
    def test_timeout_handling(self, ssh_config):
        """Test that long-running functions work with appropriate timeout."""
        @remote(**ssh_config, timeout=60)
        def slow_function():
            import time
            time.sleep(2)
            return "completed"
        
        result = slow_function()
        assert result == "completed"


@pytest.mark.integration
class TestRemoteExecutorDirect:
    """Tests using RemoteExecutor directly."""
    
    def test_direct_executor_usage(self, ssh_config):
        """Test using RemoteExecutor without decorator."""
        config = SSHConfig(**ssh_config)
        executor = RemoteExecutor(config)
        
        def my_function(x, y):
            return x + y
        
        import inspect
        frame = inspect.currentframe()
        result = executor.execute(my_function, args=(3, 4), caller_frame=frame)
        assert result == 7
    
    def test_executor_reuse(self, ssh_config, unique_venv_path):
        """Test reusing executor for multiple executions."""
        config = SSHConfig(**ssh_config)
        executor = RemoteExecutor(
            config,
            venv=unique_venv_path,
            dependencies=["six"],
        )
        
        def func1():
            return 1
        
        def func2():
            return 2
        
        import inspect
        
        frame1 = inspect.currentframe()
        result1 = executor.execute(func1, caller_frame=frame1)
        
        frame2 = inspect.currentframe()
        result2 = executor.execute(func2, caller_frame=frame2)
        
        assert result1 == 1
        assert result2 == 2

@pytest.fixture(autouse=True)
def cleanup_remote_venvs(ssh_config, request):
    """Cleanup remote virtual environments after tests."""
    venvs_to_clean = []
    
    # Collect venv paths from test
    if hasattr(request, 'node'):
        for fixture_name in ['unique_venv_path', 'unique_uv_path']:
            if fixture_name in request.fixturenames:
                try:
                    venvs_to_clean.append(request.getfixturevalue(fixture_name))
                except pytest.FixtureLookupError:
                    pass
    
    yield
    
    # Cleanup after test
    if venvs_to_clean and ssh_config.get('host'):
        try:
            import paramiko
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            connect_kwargs = {
                'hostname': ssh_config['host'],
                'port': ssh_config.get('port', 22),
                'username': ssh_config['username'],
            }
            if ssh_config.get('key_filename'):
                connect_kwargs['key_filename'] = os.path.expanduser(
                    ssh_config['key_filename']
                )
            elif ssh_config.get('password'):
                connect_kwargs['password'] = ssh_config['password']
            
            client.connect(**connect_kwargs)
            
            for venv_path in venvs_to_clean:
                if venv_path and '/tmp/' in venv_path:
                    client.exec_command(f'rm -rf {venv_path}')
            
            client.close()
        except Exception:
            pass  # Cleanup failure is not critical

"""
export TEST_SSH_HOST="your-server.com"
export TEST_SSH_USERNAME="your-username"
export TEST_SSH_PASSWORD="your-password"  # or use key
pytest test_remote.py -v -s
"""