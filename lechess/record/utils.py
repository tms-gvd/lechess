"""Utility functions for robot observation and visualization."""

import time
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import log_rerun_data
from lechess.record.config import FPS


def display_observation(robot, robot_observation_processor):
    """Display robot observation in rerun until Enter is pressed."""
    log_say("Displaying observation, press Enter to quit")
    from pynput import keyboard

    events = {"exit": False}

    def on_press(key):
        if key == keyboard.Key.enter:
            events["exit"] = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while not events["exit"]:
        start_time = time.perf_counter()
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        log_rerun_data(obs_processed)
        dt_time = time.perf_counter() - start_time
        time.sleep(max(1 / FPS - dt_time, 1 / FPS))

    listener.stop()

