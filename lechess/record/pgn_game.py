"""PGN game parsing and move access utilities."""

import io
import chess
import chess.pgn
import chess.svg
import cairosvg
from PIL import Image
import numpy as np


class PGNGame:
    """
    Provides indexed access to PGN game moves:
      - FEN before the move
      - the SAN move notation
      - PNG image bytes of the board with an arrow for the move (computed on the fly)
    Each index corresponds to one move (either white or black, depending on color_filter).
    """

    def __init__(self, pgn_path, color_filter=None):
        self.pgn_path = pgn_path
        self.color_filter = color_filter  # 'white' or 'black' or None

        with open(pgn_path, "r", encoding="utf-8") as f:
            self.game = chess.pgn.read_game(f)

        if self.game is None:
            raise ValueError("No valid PGN game found in file.")

        self.moves = list(self.game.mainline_moves())

        # Precompute fen as a list
        # Also store move objects for __getitem__
        self.all_fen = []  # List of FEN strings
        self.moves_filtered = []  # List of chess.Move objects

        board = self.game.board()

        for move in self.moves:
            fen_before = board.fen()

            # Determine whose turn it is (before the move)
            is_white_turn = board.turn == chess.WHITE

            # Filter by color if specified
            if self.color_filter is not None:
                if self.color_filter == "white" and not is_white_turn:
                    board.push(move)
                    continue
                elif self.color_filter == "black" and is_white_turn:
                    board.push(move)
                    continue

            # Store this move's data
            self.all_fen.append(fen_before)
            self.moves_filtered.append(move)

            board.push(move)

    def __len__(self):
        """Return the number of moves matching the color filter."""
        return len(self.all_fen)

    def __getitem__(self, idx):
        """
        Return (fen, move_san, image) for the move at index idx.
        The image and move_san are computed on the fly.
        """
        if idx < 0 or idx >= len(self.all_fen):
            raise IndexError(f"Index {idx} out of range for {len(self.all_fen)} moves")

        fen = self.all_fen[idx]
        move = self.moves_filtered[idx]

        # Reconstruct board state from FEN
        board = chess.Board(fen)

        # Compute move_san on the fly
        move_san = board.san(move)

        # Compute image on the fly
        png_bytes = cairosvg.svg2png(
            bytestring=chess.svg.board(
                board=board,
                arrows=[chess.svg.Arrow(move.from_square, move.to_square)],
            ).encode("utf-8")
        )
        # Convert PNG bytes to numpy array for rerun
        image = np.array(Image.open(io.BytesIO(png_bytes)))

        return fen, move_san, image

