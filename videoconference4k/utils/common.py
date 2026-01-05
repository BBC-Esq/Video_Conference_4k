import os
import re
import sys
import cv2
import types
import errno
import stat
import shutil
import importlib
import requests
import numpy as np
import logging as log
import platform
import socket
import warnings
from functools import wraps
from tqdm import tqdm
from contextlib import closing
from pathlib import Path
from colorlog import ColoredFormatter
from packaging.version import parse
from requests.adapters import HTTPAdapter, Retry
from ..version import __version__
from typing import List, Optional, Union
from numpy.typing import NDArray


def set_cuda_paths():
    venv_base = Path(sys.executable).parent.parent
    nvidia_base = venv_base / 'Lib' / 'site-packages' / 'nvidia'
    if not nvidia_base.exists():
        return
    paths_to_add = [
        str(nvidia_base / 'cuda_runtime' / 'bin'),
        str(nvidia_base / 'cuda_runtime' / 'lib' / 'x64'),
        str(nvidia_base / 'cuda_runtime' / 'include'),
        str(nvidia_base / 'cublas' / 'bin'),
        str(nvidia_base / 'cudnn' / 'bin'),
        str(nvidia_base / 'cuda_nvrtc' / 'bin'),
        str(nvidia_base / 'cuda_nvcc' / 'bin'),
    ]
    current_path = os.environ.get('PATH', '')
    os.environ['PATH'] = os.pathsep.join(paths_to_add + [current_path])
    triton_cuda_path = nvidia_base / 'cuda_runtime'
    current_cuda = os.environ.get('CUDA_PATH', '')
    os.environ['CUDA_PATH'] = os.pathsep.join([str(triton_cuda_path), current_cuda])


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


ver_is_logged = False

logger = log.getLogger("Utils")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


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
                    or f"Parameter '{parameter}' is deprecated and will be removed in future versions.",
                    DeprecationWarning,
                    stacklevel=stacklevel,
                )
            else:
                warnings.warn(
                    message
                    or f"Function '{func.__name__}' is deprecated and will be removed in future versions.",
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
            msg = """Unsupported version '{}' found. VideoConference4k requires '{}' dependency installed with version '{}' or greater. 
            Update it with  `pip install -U {}` command.""".format(
                parent_module, min_version, version, install_name
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


def check_CV_version() -> int:
    if parse(cv2.__version__) >= parse("4"):
        return 4
    else:
        return 3


def check_open_port(address: str, port: int = 22) -> bool:
    if not address:
        return False
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
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


def check_gstreamer_support(logging: bool = False) -> bool:
    raw = cv2.getBuildInformation()
    gst = [
        x.strip()
        for x in raw.split("\n")
        if x and re.search(r"GStreamer[,-:]+\s*(?:YES|NO)", x)
    ]
    if gst and "YES" in gst[0]:
        version = re.search(r"(\d+\.)?(\d+\.)?(\*|\d+)", gst[0])
        logging and logger.debug("Found GStreamer version:{}".format(version[0]))
        return version[0] >= "1.0.0"
    else:
        logger.warning("GStreamer not found!")
        return False


def check_output(*args: Union[list, tuple], **kwargs: dict) -> bytes:
    import subprocess as sp

    if platform.system() == "Windows":
        sp._cleanup = lambda: None

    retrieve_stderr = kwargs.pop("force_retrieve_stderr", False)

    process = sp.Popen(
        stdout=sp.PIPE,
        stderr=sp.DEVNULL if not (retrieve_stderr) else sp.PIPE,
        *args,
        **kwargs,
    )
    output, stderr = process.communicate()
    retcode = process.poll()

    if retcode and not (retrieve_stderr):
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = args[0]
        error = sp.CalledProcessError(retcode, cmd)
        error.output = output
        raise error

    return output if not (retrieve_stderr) else stderr


def get_supported_vencoders(path: str) -> List[str]:
    encoders = check_output([path, "-hide_banner", "-encoders"])
    splitted = encoders.split(b"\n")
    supported_vencoders = [
        x.decode("utf-8").strip()
        for x in splitted[2 : len(splitted) - 1]
        if x.decode("utf-8").strip().startswith("V")
    ]
    finder = re.compile(r"[A-Z]*[\.]+[A-Z]*\s[a-z0-9_-]*")
    outputs = finder.findall("\n".join(supported_vencoders))
    return [[s for s in o.split(" ")][-1] for o in outputs]


def get_supported_demuxers(path: str) -> List[str]:
    demuxers = check_output([path, "-hide_banner", "-demuxers"])
    splitted = [x.decode("utf-8").strip() for x in demuxers.split(b"\n")]
    split_index = [idx for idx, s in enumerate(splitted) if "--" in s][0]
    supported_demuxers = splitted[split_index + 1 : len(splitted) - 1]
    outputs = [re.search(r"\s[a-z0-9_,-]{2,}\s", d) for d in supported_demuxers]
    outputs = [o.group(0) for o in outputs if o]
    return [o.strip() if not ("," in o) else o.split(",")[-1].strip() for o in outputs]


def get_supported_pixfmts(path: str) -> List[str]:
    pxfmts = check_output([path, "-hide_banner", "-pix_fmts"])
    splitted = pxfmts.split(b"\n")
    srtindex = [i for i, s in enumerate(splitted) if b"-----" in s]
    supported_pxfmts = [
        x.decode("utf-8").strip()
        for x in splitted[srtindex[0] + 1 :]
        if x.decode("utf-8").strip()
    ]
    finder = re.compile(r"([A-Z]*[\.]+[A-Z]*\s[a-z0-9_-]*)(\s+[0-4])(\s+[0-9]+)")
    outputs = finder.findall("\n".join(supported_pxfmts))
    return [[s for s in o[0].split(" ")][-1] for o in outputs if len(o) == 3]


def is_valid_url(path: str, url: str = None, logging: bool = False) -> bool:
    if url is None or not (url):
        logger.warning("URL is empty!")
        return False
    extracted_scheme_url = url.split("://", 1)[0]
    protocols = check_output([path, "-hide_banner", "-protocols"])
    splitted = [x.decode("utf-8").strip() for x in protocols.split(b"\n")]
    supported_protocols = splitted[splitted.index("Output:") + 1 : len(splitted) - 1]
    supported_protocols += (
        ["rtsp", "rtsps"] if "rtsp" in get_supported_demuxers(path) else []
    )
    if extracted_scheme_url and extracted_scheme_url in supported_protocols:
        logging and logger.debug(
            "URL scheme `{}` is supported by FFmpeg.".format(extracted_scheme_url)
        )
        return True
    else:
        logger.warning(
            "URL scheme `{}` isn't supported by FFmpeg!".format(extracted_scheme_url)
        )
        return False


def validate_video(
    path: str, video_path: str = None, logging: bool = False
) -> Optional[dict]:
    if video_path is None or not (video_path):
        logger.warning("Video path is empty!")
        return None

    metadata = check_output(
        [path, "-hide_banner", "-i", video_path], force_retrieve_stderr=True
    )
    stripped_data = [x.decode("utf-8").strip() for x in metadata.split(b"\n")]
    logging and logger.debug(stripped_data)
    result = {}
    for data in stripped_data:
        output_a = re.findall(r"([1-9]\d+)x([1-9]\d+)", data)
        output_b = re.findall(r"\d+(?:\.\d+)?\sfps", data)
        if len(result) == 2:
            break
        if output_b and not "framerate" in result:
            result["framerate"] = re.findall(r"[\d\.\d]+", output_b[0])[0]
        if output_a and not "resolution" in result:
            result["resolution"] = output_a[-1]

    return result if (len(result) == 2) else None


def create_blank_frame(
    frame: NDArray = None, text: str = "", logging: bool = False
) -> NDArray:
    if frame is None or not (isinstance(frame, np.ndarray)):
        raise ValueError("[Utils:ERROR] :: Input frame is invalid!")
    (height, width) = frame.shape[:2]
    blank_frame = np.zeros(frame.shape, frame.dtype)
    if text and isinstance(text, str):
        logging and logger.debug("Adding text: {}".format(text))
        font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        fontScale = min(height, width) / (25 / 0.25)
        textsize = cv2.getTextSize(text, font, fontScale, 5)[0]
        textX = (width - textsize[0]) // 2
        textY = (height + textsize[1]) // 2
        cv2.putText(
            blank_frame, text, (textX, textY), font, fontScale, (125, 125, 125), 6
        )

    return blank_frame


def extract_time(value: str) -> int:
    if not (value):
        logger.warning("Value is empty!")
        return 0
    else:
        stripped_data = value.strip()
        t_duration = re.findall(r"\d{2}:\d{2}:\d{2}(?:\.\d{2})?", stripped_data)
        return (
            sum(
                float(x) * 60**i
                for i, x in enumerate(reversed(t_duration[0].split(":")))
            )
            if t_duration
            else 0
        )


def validate_audio(path: str, source: Union[str, list] = None) -> str:
    if source is None or not (source):
        logger.warning("Audio input source is empty!")
        return ""

    cmd = [path, "-hide_banner"] + (
        source if isinstance(source, list) else ["-i", source]
    )
    metadata = check_output(cmd, force_retrieve_stderr=True)
    audio_bitrate_meta = [
        line.strip()
        for line in metadata.decode("utf-8").split("\n")
        if "Audio:" in line
    ]
    audio_bitrate = (
        re.findall(r"([0-9]+)\s(kb|mb|gb)\/s", audio_bitrate_meta[0])[-1]
        if audio_bitrate_meta
        else ""
    )
    audio_samplerate_metadata = [
        line.strip()
        for line in metadata.decode("utf-8").split("\n")
        if all(x in line for x in ["Audio:", "Hz"])
    ]
    audio_samplerate = (
        re.findall(r"[0-9]+\sHz", audio_samplerate_metadata[0])[0]
        if audio_samplerate_metadata
        else ""
    )
    if audio_bitrate:
        return "{}{}".format(int(audio_bitrate[0].strip()), audio_bitrate[1].strip()[0])
    elif audio_samplerate:
        sample_rate_value = int(audio_samplerate.split(" ")[0])
        channels_value = 1 if "mono" in audio_samplerate_metadata[0] else 2
        bit_depth_value = re.findall(
            r"(u|s|f)([0-9]+)(le|be)", audio_samplerate_metadata[0]
        )[0][1]
        return (
            (
                str(
                    get_audio_bitrate(
                        sample_rate_value, channels_value, int(bit_depth_value)
                    )
                )
                + "k"
            )
            if bit_depth_value
            else ""
        )
    else:
        return ""


def get_audio_bitrate(samplerate: int, channels: int, bit_depth: float) -> int:
    return round((samplerate * channels * bit_depth) / 1000)


def get_video_bitrate(width: int, height: int, fps: float, bpp: float) -> int:
    return round((width * height * bpp * fps) / 1000)


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


def capPropId(property: str, logging: bool = True) -> int:
    integer_value = 0
    try:
        integer_value = getattr(cv2, property)
    except Exception as e:
        logging and logger.exception(str(e))
        logger.critical("`{}` is not a valid OpenCV property!".format(property))
        return None
    return integer_value


def retrieve_best_interpolation(interpolations: list) -> Optional[int]:
    if isinstance(interpolations, list):
        for intp in interpolations:
            interpolation = capPropId(intp, logging=False)
            if not (interpolation is None):
                return interpolation
    return None


def reducer(
    frame: NDArray = None,
    percentage: Union[int, float] = 0,
    interpolation: int = cv2.INTER_LANCZOS4,
) -> NDArray:
    if frame is None:
        raise ValueError("[Utils:ERROR] :: Input frame cannot be NoneType!")

    if not isinstance(frame, np.ndarray):
        raise ValueError("[Utils:ERROR] :: Input frame must be a numpy array!")

    if not (percentage > 0 and percentage < 90):
        raise ValueError(
            "[Utils:ERROR] :: Given frame-size reduction percentage is invalid, Kindly refer docs."
        )

    if not (isinstance(interpolation, int)):
        raise ValueError(
            "[Utils:ERROR] :: Given interpolation is invalid, Kindly refer docs."
        )

    (height, width) = frame.shape[:2]

    reduction = ((100 - percentage) / 100) * width
    ratio = reduction / float(width)
    dimensions = (int(reduction), int(height * ratio))

    return cv2.resize(frame, dimensions, interpolation=interpolation)


def dict2Args(param_dict: dict) -> list:
    args = []
    for key in param_dict.keys():
        if key in ["-clones"] or key.startswith("-core"):
            if isinstance(param_dict[key], list):
                args.extend(param_dict[key])
            else:
                logger.warning(
                    "{} with invalid datatype:`{}`, Skipped!".format(
                        "Core parameter" if key.startswith("-core") else "Clone",
                        param_dict[key],
                    )
                )
        else:
            args.append(key)
            args.append(str(param_dict[key]))
    return args


def get_valid_ffmpeg_path(
    custom_ffmpeg: str = "",
    is_windows: bool = False,
    ffmpeg_download_path: str = "",
    logging: bool = False,
) -> Union[str, bool]:
    final_path = ""
    if is_windows:
        if custom_ffmpeg:
            final_path += custom_ffmpeg
        else:
            try:
                if not (ffmpeg_download_path):
                    import tempfile

                    ffmpeg_download_path = tempfile.gettempdir()

                logging and logger.debug(
                    "FFmpeg Windows Download Path: {}".format(ffmpeg_download_path)
                )

                os_bit = (
                    ("win64" if platform.machine().endswith("64") else "win32")
                    if is_windows
                    else ""
                )
                _path = download_ffmpeg_binaries(
                    path=ffmpeg_download_path, os_windows=is_windows, os_bit=os_bit
                )
                final_path += _path

            except Exception as e:
                logger.exception(str(e))
                logger.error(
                    "Error in downloading FFmpeg binaries, Check your network and Try again!"
                )
                return False

        if os.path.isfile(final_path):
            pass
        elif os.path.isfile(os.path.join(final_path, "ffmpeg.exe")):
            final_path = os.path.join(final_path, "ffmpeg.exe")
        else:
            logging and logger.debug(
                "No valid FFmpeg executables found at Custom FFmpeg path!"
            )
            return False
    else:
        if custom_ffmpeg:
            if os.path.isfile(custom_ffmpeg):
                final_path += custom_ffmpeg
            elif os.path.isfile(os.path.join(custom_ffmpeg, "ffmpeg")):
                final_path = os.path.join(custom_ffmpeg, "ffmpeg")
            else:
                logging and logger.debug(
                    "No valid FFmpeg executables found at Custom FFmpeg path!"
                )
                return False
        else:
            final_path += "ffmpeg"

    logging and logger.debug("Final FFmpeg Path: {}".format(final_path))

    return final_path if validate_ffmpeg(final_path, logging=logging) else False


def download_ffmpeg_binaries(
    path: str, os_windows: bool = False, os_bit: str = ""
) -> str:
    final_path = ""
    if os_windows and os_bit:
        file_url = "https://github.com/abhiTronix/FFmpeg-Builds/releases/latest/download/ffmpeg-static-{}-gpl.zip".format(
            os_bit
        )

        file_name = os.path.join(
            os.path.abspath(path), "ffmpeg-static-{}-gpl.zip".format(os_bit)
        )
        file_path = os.path.join(
            os.path.abspath(path),
            "ffmpeg-static-{}-gpl/bin/ffmpeg.exe".format(os_bit),
        )
        base_path, _ = os.path.split(file_name)
        if os.path.isfile(file_path):
            final_path += file_path
        else:
            import zipfile

            assert os.access(path, os.W_OK), (
                "[Utils:ERROR] :: Permission Denied, Cannot write binaries to directory = "
                + path
            )
            os.path.isfile(file_name) and delete_file_safe(file_name)
            with open(file_name, "wb") as f:
                logger.debug(
                    "No Custom FFmpeg path provided. Auto-Installing FFmpeg static binaries from GitHub Mirror now. Please wait..."
                )
                with requests.Session() as http:
                    retries = Retry(
                        total=3,
                        backoff_factor=1,
                        status_forcelist=[429, 500, 502, 503, 504],
                    )
                    adapter = TimeoutHTTPAdapter(timeout=2.0, max_retries=retries)
                    http.mount("https://", adapter)
                    response = http.get(file_url, stream=True)
                    response.raise_for_status()
                    total_length = (
                        response.headers.get("content-length")
                        if "content-length" in response.headers
                        else len(response.content)
                    )
                    assert not (
                        total_length is None
                    ), "[Utils:ERROR] :: Failed to retrieve files, check your Internet connectivity!"
                    bar = tqdm(total=int(total_length), unit="B", unit_scale=True)
                    for data in response.iter_content(chunk_size=4096):
                        f.write(data)
                        len(data) > 0 and bar.update(len(data))
                    bar.close()
            logger.debug("Extracting executables.")
            with zipfile.ZipFile(file_name, "r") as zip_ref:
                zip_fname, _ = os.path.split(zip_ref.infolist()[0].filename)
                zip_ref.extractall(base_path)
            delete_file_safe(file_name)
            logger.debug("FFmpeg binaries for Windows configured successfully!")
            final_path += file_path
    return final_path


def validate_ffmpeg(path: str, logging: bool = False) -> bool:
    try:
        version = check_output([path, "-version"])
        firstline = version.split(b"\n")[0]
        version = firstline.split(b" ")[2].strip()
        logging and logger.info("FFmpeg validity Test Passed!")
        logging and logger.debug(
            "Found valid FFmpeg Version: `{}` installed on this system".format(version)
        )
    except Exception as e:
        logging and logger.exception(str(e))
        logger.error("FFmpeg validity Test Failed!")
        return False
    return True


def generate_auth_certificates(
    path: str, overwrite: bool = False, logging: bool = False
) -> tuple:
    import zmq.auth

    if os.path.basename(path) != ".videoconference4k":
        path = os.path.join(path, ".videoconference4k")

    keys_dir = os.path.join(path, "keys")
    mkdir_safe(keys_dir, logging=logging)

    public_keys_dir = os.path.join(keys_dir, "public_keys")
    secret_keys_dir = os.path.join(keys_dir, "private_keys")

    if overwrite:
        for dirs in [public_keys_dir, secret_keys_dir]:
            if os.path.exists(dirs):
                shutil.rmtree(dirs)
            mkdir_safe(dirs, logging=logging)

        server_public_file, server_secret_file = zmq.auth.create_certificates(
            keys_dir, "server"
        )
        client_public_file, client_secret_file = zmq.auth.create_certificates(
            keys_dir, "client"
        )

        for key_file in os.listdir(keys_dir):
            if key_file.endswith(".key"):
                shutil.move(os.path.join(keys_dir, key_file), public_keys_dir)
            elif key_file.endswith(".key_secret"):
                shutil.move(os.path.join(keys_dir, key_file), secret_keys_dir)
            else:
                redundant_key = os.path.join(keys_dir, key_file)
                if os.path.isfile(redundant_key):
                    delete_file_safe(redundant_key)
    else:
        status_public_keys = validate_auth_keys(public_keys_dir, ".key")
        status_private_keys = validate_auth_keys(secret_keys_dir, ".key_secret")

        if status_private_keys and status_public_keys:
            return (keys_dir, secret_keys_dir, public_keys_dir)

        if not (status_public_keys):
            mkdir_safe(public_keys_dir, logging=logging)

        if not (status_private_keys):
            mkdir_safe(secret_keys_dir, logging=logging)

        server_public_file, server_secret_file = zmq.auth.create_certificates(
            keys_dir, "server"
        )
        client_public_file, client_secret_file = zmq.auth.create_certificates(
            keys_dir, "client"
        )

        for key_file in os.listdir(keys_dir):
            if key_file.endswith(".key") and not (status_public_keys):
                shutil.move(
                    os.path.join(keys_dir, key_file), os.path.join(public_keys_dir, ".")
                )
            elif key_file.endswith(".key_secret") and not (status_private_keys):
                shutil.move(
                    os.path.join(keys_dir, key_file), os.path.join(secret_keys_dir, ".")
                )
            else:
                redundant_key = os.path.join(keys_dir, key_file)
                if os.path.isfile(redundant_key):
                    delete_file_safe(redundant_key)

    status_public_keys = validate_auth_keys(public_keys_dir, ".key")
    status_private_keys = validate_auth_keys(secret_keys_dir, ".key_secret")

    if not (status_private_keys) or not (status_public_keys):
        raise RuntimeError(
            "[Utils:ERROR] :: Unable to generate valid ZMQ authentication certificates at `{}`!".format(
                keys_dir
            )
        )

    return (keys_dir, secret_keys_dir, public_keys_dir)


def validate_auth_keys(path: str, extension: str) -> bool:
    if not (os.path.exists(path)):
        return False

    if not (os.listdir(path)):
        return False

    keys_buffer = []

    for key_file in os.listdir(path):
        key = os.path.splitext(key_file)
        if key and (key[0] in ["server", "client"]) and (key[1] == extension):
            keys_buffer.append(key_file)

    len(keys_buffer) == 1 and delete_file_safe(os.path.join(path, keys_buffer[0]))

    return True if (len(keys_buffer) == 2) else False