import os
import shutil

from .common import (
    get_logger,
    mkdir_safe,
    delete_file_safe,
)

logger = get_logger("AuthUtils")


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
            "[AuthUtils:ERROR] :: Unable to generate valid ZMQ authentication certificates at `{}`!".format(
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