import argparse
import os
import shutil
import sys
import termios
import time
import rerun as rr
import rerun.blueprint as rrb
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lechess.record import (
    BASE_DIR,
    BATCH_ENCODING_SIZE,
    EPISODE_TIME_SEC,
    FPS,
    ID_FOLLOWER,
    ID_LEADER,
    PLAY_SOUNDS,
    PORT_FOLLOWER,
    PORT_LEADER,
)
from lechess.record import PGNGame
from lechess.record import display_observation


def main():
    parser = argparse.ArgumentParser(description="Record robot teleoperation for chess moves")
    parser.add_argument("--pgn_path", type=str, required=True, help="Path to PGN file")
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace dataset repo ID")
    parser.add_argument(
        "--color",
        type=str,
        choices=["white", "black"],
        required=True,
        help="Color to record moves for",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Push to HuggingFace hub")
    args = parser.parse_args()

    pgn_path = args.pgn_path
    hf_repo_id = args.repo_id
    color = args.color

    dataset_dir = os.path.join(BASE_DIR, hf_repo_id)

    # Check if dataset exists
    if os.path.exists(dataset_dir):
        print(f"Dataset already exists in {dataset_dir}")
        answer = input("Delete it? (y/n)")
        if answer == "y":
            shutil.rmtree(dataset_dir)
            print(f"Dataset deleted")
        else:
            print("Dataset not deleted")
            exit()

    # Initialize logging
    init_logging()

    # Load PGN game and filter moves by color
    pgn_game = PGNGame(pgn_path, color_filter=color)

    if len(pgn_game) == 0:
        print(f"No moves found for color '{color}' in the PGN file.")
        exit()

    print(f"Found {len(pgn_game)} moves for {color}")

    # Create the robot and teleoperator configurations
    camera_config = {
        "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS),
        "top": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=FPS),
    }
    robot_config = SO101FollowerConfig(port=PORT_FOLLOWER, id=ID_FOLLOWER, cameras=camera_config)
    teleop_config = SO101LeaderConfig(port=PORT_LEADER, id=ID_LEADER)

    # Initialize the robot and teleoperator
    robot = SO101Follower(robot_config)
    teleop = SO101Leader(teleop_config)

    # Create default processor pipelines
    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    # Configure the dataset features using pipeline-based approach
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    # Create the dataset
    dataset = LeRobotDataset.create(
        repo_id=hf_repo_id,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
        batch_encoding_size=BATCH_ENCODING_SIZE,
    )

    # Initialize the keyboard listener and rerun visualization
    listener, events = init_keyboard_listener()
    init_rerun(session_name="recording")
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="chessboard"),
            rrb.Spatial2DView(origin="observation.front"),
            rrb.Spatial2DView(origin="observation.top"),
            column_shares=[30, 35, 35],
        )
    )
    rr.send_blueprint(blueprint)

    # Connect the robot and teleoperator
    robot.connect()
    teleop.connect()

    # Main recording loop with VideoEncodingManager
    with VideoEncodingManager(dataset):
        recorded_episodes = 0
        episode_task = None  # Store task for re-recording
        move_idx = 0

        # Display observation to check if the scene is set up correctly
        display_observation(robot, robot_observation_processor)

        while move_idx < len(pgn_game) and not events["stop_recording"]:
            fen, move_san, image = pgn_game[move_idx]

            # Show the chess position
            print(f"\n{'='*60}")
            print(f"Move {move_idx + 1}/{len(pgn_game)}: {move_san}")
            print(f"FEN: {fen}")
            print(f"{'='*60}")
            rr.log("chessboard", rr.Image(image))

            # Wait for user to press "g" to record
            print(
                "\nPress 'g' to start recording, 'w' for next move, 'b' for previous move, or 'q' to quit..."
            )

            # Flush stdin to clear any leftover characters from keyboard listener
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)

            while True:
                user_input = input().strip().lower()
                if user_input == "g":
                    break
                elif user_input == "w":
                    break
                elif user_input == "b":
                    break
                elif user_input == "q":
                    events["stop_recording"] = True
                    break
                else:
                    print(
                        "Invalid input. Press 'g' to record, 'w' for next, 'b' for previous, or 'q' to quit."
                    )

            if events["stop_recording"]:
                break

            # Handle navigation - go to next or previous move without recording
            if user_input == "w":
                move_idx += 1
                if move_idx >= len(pgn_game):
                    print("Already at the last move.")
                    move_idx = len(pgn_game) - 1
                continue
            elif user_input == "b":
                move_idx -= 1
                if move_idx < 0:
                    print("Already at the first move.")
                    move_idx = 0
                continue

            # Set task description automatically
            episode_task = f"FEN: {fen} $$ MOVE: {move_san}"
            print(f"Recording move {move_idx + 1} with task:")
            print(episode_task)

            log_say(f"Recording move {move_idx + 1} of {len(pgn_game)}", PLAY_SOUNDS)

            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=episode_task,
                display_data=True,
            )

            if events["rerecord_episode"]:
                log_say("Re-record episode", PLAY_SOUNDS)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                # Keep the same episode_task for re-recording, don't increment move_idx
                continue

            dataset.save_episode()
            recorded_episodes += 1

            # Every 5 moves, remind user to adjust lighting and chessboard position
            if recorded_episodes % 5 == 0:
                log_say("Please modify the lighting and chessboard position", PLAY_SOUNDS)
                time.sleep(2)
                display_observation(robot, robot_observation_processor)

            episode_task = None  # Reset task for next episode
            move_idx += 1

    # Clean up
    log_say("Stop recording", PLAY_SOUNDS, blocking=True)
    robot.disconnect()
    teleop.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    if args.push_to_hub:
        # Copy PGN file to dataset directory (will be uploaded by push_to_hub)
        os.makedirs(dataset_dir, exist_ok=True)
        pgn_filename = os.path.basename(pgn_path)
        dest_pgn_path = os.path.join(dataset_dir, pgn_filename)
        shutil.copy2(pgn_path, dest_pgn_path)
        print(f"Copied PGN file to {dest_pgn_path}")

        # Push to hub with chess-specific information via card_kwargs
        dataset.push_to_hub(
            tags=["chess"],
            chess_pgn_file=pgn_filename,
            chess_color=color,
            chess_total_episodes=recorded_episodes,
        )
        print(f"Pushed dataset to hub with chess information")


if __name__ == "__main__":
    main()
