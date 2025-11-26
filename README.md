# LeChess

A robotics project for recording chess move demonstrations using SO-101 robot arms and train a policy to reproduce any chess move.

## Installation

*Tested on MacBook Air M1*

Clone this repo with `lerobot` submodule:
```
git clone --recursive <repo>
```
It uses a fork of `lerobot` that is up-to-date with the original one, with slight modifications.

### Environment setup

Follow [lerobot](https://github.com/huggingface/lerobot#environment-setup) instructions to setup a conda environment with python 3.10 and `ffmpeg`:
```
conda create -y -n lechess python=3.10
conda activate lechess
conda install ffmpeg -c conda-forge
```

### Install LeRobot from source

Install the library in editable mode with `feetech` to control the SO101:
```
cd lerobot && pip install -e ".[feetech]" && cd ..
```

Install the additional packages required for LeChess:
```bash
pip install -r requirements.txt
```

### Setup the hardware

You need a SO101 leader and follower arm + two `OpenCVCamera` cameras to reproduce our work.
Note that you can also use a S0100 leader and follower but you need to calibrate them as `SO101Leader` and `SO101Follower`.

We are using slightly modded version of the arms, have a look at our [hardware setup](docs/hardware.md) for more details.

### Environment variable setup

Create a `.env` file in the project root directory with the following required variables:
```bash
# Robot IDs
ID_FOLLOWER=<your_follower_robot_id>
ID_LEADER=<your_leader_robot_id>

# Serial ports
PORT_FOLLOWER=<serial_port_for_follower>
PORT_LEADER=<serial_port_for_leader>
```

### Login to HuggingFace CLI

Generate a [User access token](https://huggingface.co/docs/hub/security-tokens) with write permission and log in:
```
hf auth login --token <your-access-token>
hf auth whoami
```
This is required to push the recorded to [HF datasets hub](https://huggingface.co/datasets).

## Teleoperate

Run `bash teleop.sh` that uses the variables from `.env`

Visualization is provided with `rerun`.

## Record

`record.py` is the main script for recording robot teleoperation demonstrations of chess moves. It enables you to:

- **Load PGN files**: Parse chess games from PGN (Portable Game Notation) files. You can find some PGN files in `./chess_games/`
- **Display chess positions**: Visualize the current board state with move arrows indicating the move to be executed
- **Record robot demonstrations**: Capture teleoperated robot movements for each chess move
- **Create LeRobot datasets**: Save recordings in the LeRobot dataset format for training
- **Upload to HuggingFace Hub**: Optionally push datasets to the HuggingFace hub with chess-specific metadata

### Usage

```bash
python record.py --pgn_path <path_to_pgn_file> --repo_id <huggingface_repo_id> --color <white|black> [--push_to_hub]
```

**Arguments:**
- `--pgn_path`: Path to the PGN file containing the chess game
- `--repo_id`: HuggingFace dataset repository ID (e.g., `username/chess_dataset`)
- `--color`: Color to record moves for (`white` or `black`)
- `--push_to_hub`: (Optional) Push the dataset to HuggingFace hub after recording

**Example:**
```bash
python record.py \
  --pgn_path chess_games/fischer_spassky_1972.pgn \
  --repo_id myusername/chess_fischer_spassky \
  --color white \
  --push_to_hub
```

### Workflow

1. **Initialization**: The script loads the PGN file and filters moves by the specified color (white or black)
2. **Setup check**: Displays the current robot observation to verify camera setup
3. **Move-by-move recording**:
   - Displays the chess position with the move arrow
   - Shows FEN notation and SAN move notation
   - Waits for user input:
     - `g`: Start recording the current move
     - `w`: Navigate to next move without recording
     - `b`: Navigate to previous move without recording
     - `q`: Quit recording
4. **Recording**: Captures robot teleoperation for the specified duration (default: 60 seconds). Press `right arrow` to exit before the end
5. **Re-recording**: Press `left arrow` during recording to re-record the current move
6. **Periodic checks**: Every 5 moves, prompts to adjust lighting and chessboard position
7. **Dataset creation**: Saves each recorded episode with metadata including FEN and move notation
8. **Upload**: Optionally pushes the complete dataset to HuggingFace hub with chess-specific tags

## Acknowledgment

- The LeRobot team [huggingface/lerobot](https://github.com/huggingface/lerobot) for LeRobot, teleoperation, recording, `LeRobotDataset` and so much else
- [Chojins](https://github.com/Chojins) for its work and ideas on LeRobot playing chess: 3D printed magnetic chess set, pedestal, gripper, etc