import os
import sys
import types
import errno
import stat
import importlib
import logging as log
import socket
import warnings
from functools import wraps
from pathlib import Path
from colorlog import ColoredFormatter
from packaging.version import parse
from requests.adapters import HTTPAdapter
from typing import Optional
from ..version import __version__


def raise_timer_resolution(period_ms: int = 1) -> bool:
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes
        return ctypes.windll.winmm.timeBeginPeriod(int(period_ms)) == 0
    except Exception:
        return False


def restore_timer_resolution(period_ms: int = 1) -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        ctypes.windll.winmm.timeEndPeriod(int(period_ms))
    except Exception:
        pass


def set_cuda_paths():
    from pathlib import Path

    venv_base = Path(sys.executable).parent.parent
    nvidia_base = venv_base / 'Lib' / 'site-packages' / 'nvidia'
    if not nvidia_base.exists():
        return

    paths_to_add = [
        nvidia_base / 'cuda_runtime' / 'bin',
        nvidia_base / 'cublas' / 'bin',
        nvidia_base / 'cuda_nvrtc' / 'bin',
    ]

    paths_to_add = [str(p) for p in paths_to_add if p.exists()]

    current_path = os.environ.get('PATH', '')
    os.environ['PATH'] = os.pathsep.join(paths_to_add + [current_path])

    if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
        for path in paths_to_add:
            try:
                os.add_dll_directory(path)
            except OSError:
                pass


set_cuda_paths()


def logger_handler():
    formatter = ColoredFormatter(
        "{green}{asctime}{reset} :: {bold_purple}{name:^13}{reset} :: {log_color}{levelname:^8}{reset} :: {bold_white}{message}",
        datefmt="%H:%M:%S",
        reset=True,
        log_colors={
            "INFO": "bold_cyan",
            "DEBUG": "bold_yellow",
            "WARNING": "bold_red,fg_thin_yellow",
            "ERROR": "bold_red",
            "CRITICAL": "bold_red,bg_white",
        },
        style="{",
    )
    file_mode = os.environ.get("VC4K_LOGFILE", False)
    handler = log.StreamHandler()
    if file_mode and isinstance(file_mode, str):
        file_path = os.path.abspath(file_mode)
        if (os.name == "nt" or os.access in os.supports_effective_ids) and os.access(
            os.path.dirname(file_path), os.W_OK
        ):
            file_path = (
                os.path.join(file_path, "videoconference4k.log")
                if os.path.isdir(file_path)
                else file_path
            )
            handler = log.FileHandler(file_path, mode="a")
            formatter = log.Formatter(
                "{asctime} :: {name} :: {levelname} :: {message}",
                datefmt="%H:%M:%S",
                style="{",
            )

    handler.setFormatter(formatter)
    return handler


def get_logger(name: str) -> log.Logger:
    logger = log.getLogger(name)
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(logger_handler())
    logger.setLevel(log.DEBUG)
    return logger


ver_is_logged = False

logger = get_logger("Utils")


def log_version(logging=False):
    global ver_is_logged
    logging and not (ver_is_logged) and logger.info(
        "Running VideoConference4k Version: {}".format(str(__version__))
    )
    if logging and not (ver_is_logged):
        ver_is_logged = True


def get_module_version(module=None):
    assert not (module is None) and isinstance(
        module, types.ModuleType
    ), "[VideoConference4k:ERROR] :: Invalid module!"

    version = getattr(module, "__version__", None)
    if version is None:
        version = getattr(module, "__VERSION__", None)
    if version is None:
        raise ImportError(
            "[VideoConference4k:ERROR] ::  Can't determine version for module: `{}`!".format(
                module.__name__
            )
        )
    return str(version)


def deprecated(parameter=None, message=None, stacklevel=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if parameter and parameter in kwargs:
                warnings.warn(
                    message
                    or "Parameter '{}' is deprecated and will be removed in future versions.".format(parameter),
                    DeprecationWarning,
                    stacklevel=stacklevel,
                )
            else:
                warnings.warn(
                    message
                    or "Function '{}' is deprecated and will be removed in future versions.".format(func.__name__),
                    DeprecationWarning,
                    stacklevel=stacklevel,
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def import_dependency_safe(
    name,
    error="raise",
    pkg_name=None,
    min_version=None,
    custom_message=None,
):
    sub_class = ""
    if not name or not isinstance(name, str):
        return None
    else:
        name = name.strip()
        if name.startswith("from"):
            name = name.split(" ")
            name, sub_class = (name[1].strip(), name[-1].strip())

    assert error in [
        "raise",
        "log",
        "silent",
    ], "[VideoConference4k:ERROR] :: Invalid value at `error` parameter."

    install_name = pkg_name if not (pkg_name is None) else name

    msg = (
        custom_message
        if not (custom_message is None)
        else "Failed to find required dependency '{}'. Install it with  `pip install {}` command.".format(
            name, install_name
        )
    )
    try:
        module = importlib.import_module(name)
        module = getattr(module, sub_class) if sub_class else module
    except Exception as e:
        if error == "raise":
            if isinstance(e, ModuleNotFoundError):
                raise ModuleNotFoundError(msg) from None
            else:
                raise ImportError(msg) from e
        elif error == "log":
            logger.error(msg, exc_info=sys.exc_info())
            return None
        else:
            return None

    if not (min_version) is None:
        parent_module = name.split(".")[0]
        if parent_module != name:
            module_to_get = sys.modules[parent_module]
        else:
            module_to_get = module
        version = get_module_version(module_to_get)
        if parse(version) < parse(min_version):
            msg = "Unsupported version '{}' found. VideoConference4k requires '{}' dependency installed with version '{}' or greater. Update it with `pip install -U {}` command.".format(
                version, parent_module, min_version, install_name
            )
            if error == "silent":
                return None
            else:
                raise ImportError(msg)

    return module


DEFAULT_TIMEOUT = 3


class TimeoutHTTPAdapter(HTTPAdapter):

    def __init__(self, *args, **kwargs):
        self.timeout = DEFAULT_TIMEOUT
        if "timeout" in kwargs:
            self.timeout = kwargs["timeout"]
            del kwargs["timeout"]
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        timeout = kwargs.get("timeout")
        if timeout is None:
            kwargs["timeout"] = self.timeout
        return super().send(request, **kwargs)


def check_open_port(address: str, port: int = 22) -> bool:
    if not address:
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex((address, port)) == 0:
            return True
        else:
            return False


def check_WriteAccess(
    path: str, is_windows: bool = False, logging: bool = False
) -> bool:
    dirpath = Path(path)
    try:
        if not (dirpath.exists() and dirpath.is_dir()):
            logger.warning(
                "Specified directory `{}` doesn't exists or valid.".format(path)
            )
            return False
        else:
            path = dirpath.resolve()
    except:
        return False
    if not is_windows:
        uid = os.geteuid()
        gid = os.getegid()
        s = os.stat(path)
        mode = s[stat.ST_MODE]
        return (
            ((s[stat.ST_UID] == uid) and (mode & stat.S_IWUSR))
            or ((s[stat.ST_GID] == gid) and (mode & stat.S_IWGRP))
            or (mode & stat.S_IWOTH)
        )
    else:
        write_accessible = False
        temp_fname = os.path.join(path, "temp.tmp")
        try:
            fd = os.open(temp_fname, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
            os.close(fd)
            write_accessible = True
        except Exception as e:
            if isinstance(e, PermissionError):
                logger.error(
                    "You don't have adequate access rights to use `{}` directory!".format(
                        path
                    )
                )
            logging and logger.exception(str(e))
        finally:
            delete_file_safe(temp_fname)
        return write_accessible


def delete_file_safe(file_path: str) -> None:
    try:
        dfile = Path(file_path)
        dfile.unlink(missing_ok=True)
    except Exception as e:
        logger.exception(str(e))


def mkdir_safe(dir_path: str, logging: bool = False) -> None:
    try:
        os.makedirs(dir_path)
        logging and logger.debug("Created directory at `{}`".format(dir_path))
    except (OSError, IOError) as e:
        if e.errno != errno.EACCES and e.errno != errno.EEXIST:
            raise


def delete_ext_safe(
    dir_path: str, extensions: list = [], logging: bool = False
) -> None:
    if not extensions or not os.path.exists(dir_path):
        logger.warning("Invalid input provided for deleting!")
        return

    logger.critical("Clearing Assets at `{}`!".format(dir_path))

    for ext in extensions:
        if len(ext) == 2:
            files_ext = [
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if f.startswith(ext[0]) and f.endswith(ext[1])
            ]
        else:
            files_ext = [
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if f.endswith(ext)
            ]
        for file in files_ext:
            delete_file_safe(file)
            logging and logger.debug("Deleted file: `{}`".format(file))