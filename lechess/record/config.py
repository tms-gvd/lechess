"""Configuration and environment variable management."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def _get_env_variable(variable_name):
    """Get an environment variable or raise an error if not set."""
    if not os.getenv(variable_name):
        raise ValueError(f"{variable_name} is not set to environment variable")
    else:
        return os.getenv(variable_name)


# Configuration parameters
FPS = 30
EPISODE_TIME_SEC = 60
PLAY_SOUNDS = True
BATCH_ENCODING_SIZE = 1

ID_FOLLOWER = _get_env_variable("ID_FOLLOWER")
ID_LEADER = _get_env_variable("ID_LEADER")
PORT_FOLLOWER = _get_env_variable("PORT_FOLLOWER")
PORT_LEADER = _get_env_variable("PORT_LEADER")

# hardcoded but it is best to use the default HuggingFace datasets storage location
BASE_DIR = os.path.expanduser("~/.cache/huggingface/lerobot")

