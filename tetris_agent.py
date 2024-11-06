import numpy as np
from tetris_env import TetrisEnv
import os

class TetrisAgent:
    def __init__(self, env: TetrisEnv, is_debug: bool = False):
        self.env: TetrisEnv = env
        self.is_debug = is_debug
        self.weights = [0.42657453, 1.17615849, -0.0422209, -0.82640537]

    def evaluate_position(self, board, piece, position):
        """Evaluate a specific position and return a score based on normalized reward criteria."""
        temp_board = np.copy(board)
        temp_position = position.copy()

        # Place the piece on the temp board at the specified position
        for row, line in enumerate(piece):
            for col, cell in enumerate(line):
                if cell:
                    x, y = temp_position[0] + row, temp_position[1] + col
                    if 0 <= x < temp_board.shape[0] and 0 <= y < temp_board.shape[1]:
                        temp_board[x, y] = cell

        # Calculate reward criteria
        lines_cleared = self.count_cleared_lines(temp_board)
        holes = self.count_holes(temp_board)
        height = temp_position[0]
        empty_pillars = self.count_empty_pillars(temp_board)
        
        # Normalize the reward components
        normalized_lines_cleared = min(lines_cleared / 4, 1)  # Assume max 4 lines can be cleared at once
        normalized_holes = min(holes / (self.env.board_width * self.env.board_height), 1)  # Max possible holes
        normalized_height = height / self.env.board_height  # Normalize height to a range of [0, 1]
        empty_pillars_penalty = min(empty_pillars / 5, 1) if empty_pillars >= 2 else 0 # normalize if empty pillars is more than 2




        # Reward function with normalized components
        reward = (
            (normalized_lines_cleared ** 2) * self.weights[0]
            - normalized_holes * self.weights[1]
            - (normalized_height ** 2) * self.weights[2]
            + empty_pillars_penalty * self.weights[3]
        )

        # print(self.env.current_piece_name)
        if self.env.current_piece_name == 'L' or self.env.current_piece_name == 'J':
            # print('yep')
            try:
                if (piece == [[1,1],[0,1],[0,1]] or piece == [[1,1],[1,0],[1,0]]) and holes > 0:
                    reward = -1000
                    # print('reward = -1000')
            except:
                pass
        return reward

    def count_cleared_lines(self, board):
        """Count how many lines would be cleared in this board state."""
        return sum(np.all(row) for row in board)

    def count_holes(self, board):
        """Count holes in the board (empty cells with filled cells above them)."""
        holes = 0
        for col in range(board.shape[1]):
            found_block = False
            for row in range(board.shape[0]):
                if board[row, col] == 1:
                    found_block = True
                elif found_block and board[row, col] == 0:
                    holes += 1
        return holes

    def count_empty_pillars(self, board):
        """Count columns that have adjacent columns with at least 3 rows higher, indicating an empty pillar."""
        empty_pillars = 0

        # Calculate heights of each column
        column_heights = []
        for col in range(board.shape[1]):
            height = next((row for row in range(board.shape[0]) if board[row, col] != 0), board.shape[0])
            column_heights.append(board.shape[0] - height)

        # Check for empty pillars based on adjacent column heights
        for col in range(board.shape[1]):
            current_height = column_heights[col]

            # Check if the left neighbor exists
            if col > 0 and column_heights[col - 1] - current_height >= 3:
                empty_pillars += 1
                continue  # No need to check the right neighbor if already identified as empty

            # Check the right neighbor if it exists
            if col < board.shape[1] - 1 and column_heights[col + 1] - current_height >= 3:
                empty_pillars += 1

        return empty_pillars

    def choose_best_move(self):
        """Evaluate all possible positions for the current piece and choose the best one."""
        best_score = -float('inf')
        best_position = None
        best_rotation = 0

        # Get the current piece in stable 4x4 grid
        initial_piece = self.env.current_piece.copy()
        initial_position = self.env.current_piece.copy()

        # Explore all rotations (0, 90, 180, 270 degrees)
        for rotation in range(4):
            piece = np.rot90(initial_piece.copy(), -rotation)

            piece_width = piece.shape[1]

            # Try each column position where the piece could land
            for col in range(-1, self.env.board_width):
                # Drop the piece to the bottom in this column
                row = 0
                while self.is_valid_position(piece, (row, col)):
                    row += 1
                row -= 1  # Step back to last valid position

                # Evaluate this position
                score = self.evaluate_position(self.env.board, piece, [row, col])
                if self.is_debug:
                    self.env.render(given_piece=piece, given_position=[row, col])
                if score > best_score:
                    best_score = score
                    best_position = [row, col]
                    best_rotation = rotation

        # Return the best position and rotation
        return best_position, best_rotation

    def is_valid_position(self, piece, position):
        """Check if the piece can be placed in the given position on the board."""
        for row, line in enumerate(piece):
            for col, cell in enumerate(line):
                if cell:
                    x, y = position[0] + row, position[1] + col
                    if x >= self.env.board_height or y < 0 or y >= self.env.board_width or (x >= 0 and self.env.board[x, y]):
                        return False
        return True

    def execute_best_move(self, render:bool = False):
        """Move the piece to the chosen best position and rotation."""
        best_position, best_rotation = self.choose_best_move()
        
        # Rotate the piece to the optimal orientation
        done = False
        for _ in range(best_rotation):
            state, done = self.env.step(2)
            if render:
                self.env.render()
            if done == True:
                break

        
        # Move to the target column
        while self.env.current_position[1] > best_position[1] and done == False:
            state, done = self.env.step(0) # left
            if render:
                self.env.render()
        while self.env.current_position[1] < best_position[1] and done == False:
            state, done = self.env.step(1) # right
            if render:
                self.env.render()

        return done