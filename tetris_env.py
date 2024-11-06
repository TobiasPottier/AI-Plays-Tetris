import random
import gym
from gym import spaces
import numpy as np
import pygame
import time

class TetrisEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, board_width=10, board_height=20, time_delay=0.0):
        super(TetrisEnv, self).__init__()

        self.new_piece_spawned = False

        self.time_delay = time_delay

        self.lines_cleared_count = 0
        
        self.board_width = board_width
        self.board_height = board_height
        self.board = np.zeros((board_height, board_width), dtype=int)
        
        # Defining the pieces
        self.pieces = {
            'I': [[1, 1, 1, 1]],
            'J': [[1, 0, 0], [1, 1, 1]],
            'L': [[0, 0, 1], [1, 1, 1]],
            'O': [[1, 1], [1, 1]],
            'S': [[0, 1, 1], [1, 1, 0]],
            'T': [[0, 1, 0], [1, 1, 1]],
            'Z': [[1, 1, 0], [0, 1, 1]]
        }

        self.pieces_colors = {
            'I': (110, 236, 238),
            'J': (0, 0, 230),
            'L': (228, 164, 57),
            'O': (240, 240, 79),
            'S': (110, 236, 71),
            'T': (146, 28, 231),
            'Z': (220, 47, 33)
        }
        
        self.current_piece = None
        self.current_piece_name = None
        self.current_position = None
        self.current_piece_color = (0, 0, 0)
        self.screen = None
        self.cell_size = 30
        
        self.action_space = spaces.Discrete(3)  # 0: Left, 1: Right, 2: Rotate
        self.observation_space = spaces.Box(low=0, high=1, shape=(board_height, board_width), dtype=np.int32)

        self.reset()

    def reset(self):
        self.lines_cleared_count = 0
        self.board.fill(0)
        self.spawn_piece()
        return self.get_state()
    
    
    def get_piece_in_stable_grid(self):
        """Return a stable 4x4 grid containing the current piece centered."""
        # Initialize a 4x4 grid of zeros
        stable_grid = np.zeros((4, 4), dtype=int)
        
        # Get dimensions of the current piece
        piece_height, piece_width = self.current_piece.shape
        
        # Calculate the starting position to center the piece in the 4x4 grid
        start_row = (4 - piece_height) // 2
        start_col = (4 - piece_width) // 2
        
        # Place the piece in the center of the 4x4 grid
        stable_grid[start_row:start_row + piece_height, start_col:start_col + piece_width] = self.current_piece
        
        return stable_grid
        
    def step(self, action):
        done = False

        # Handle horizontal movement and rotation first
        if action == 0:
            self.move_left()
        elif action == 1:
            self.move_right()
        elif action == 2:
            self.rotate()

        # Always drop the piece by one space each step
        self.current_position[0] += 1

        # Check if the piece has landed
        if not self.is_valid_position():
            # Move piece back up
            self.current_position[0] -= 1
            # Lock the piece in place (current board + our current piece)
            state = np.copy(self.board)
            for row, line in enumerate(self.current_piece):
                for col, cell in enumerate(line):
                    if cell:
                        x, y = self.current_position[0] + row, self.current_position[1] + col
                        if 0 <= x < self.board_height and 0 <= y < self.board_width:
                            state[x, y] = cell
            self.board = state
            # Clear any full lines
            lines_cleared = self.clear_lines()
            # Spawn new piece
            self.spawn_piece()
            # Check if game is over
            if not self.is_valid_position():
                done = True

        return self.get_state(), done
    
    def clear_lines(self):
        """Clear full lines and make pieces above fall down.
        Returns the number of lines cleared."""
        lines_cleared = 0
        # Check each line from bottom to top
        y = self.board_height - 1
        while y >= 0:
            # If line is full (no zeros)
            if np.all(self.board[y]):
                self.lines_cleared_count += 1
                lines_cleared += 1
                # Move all lines above down
                self.board[1:y + 1] = self.board[0:y]
                # Clear top line
                self.board[0].fill(0)
            else:
                y -= 1
        
        return lines_cleared
    
    def count_holes(self):
        """Count the number of holes (empty cells with filled cells above them)"""
        holes = 0
        # For each column
        for col in range(self.board_width):
            # Find the highest block in this column
            highest_block = -1
            for row in range(self.board_height):
                if self.board[row, col]:
                    highest_block = row
                    break
            # If we found a block, count empty cells below it
            if highest_block != -1:
                holes += np.sum(self.board[highest_block:, col] == 0)
        return holes
    
    def spawn_piece(self):
        # Randomly select a piece
        self.new_piece_spawned = True
        piece_name = random.choice(list(self.pieces.keys()))
        self.current_piece_name = piece_name
        self.current_piece_color = self.pieces_colors[piece_name]
        self.current_piece = np.array(self.pieces[piece_name])
        self.current_position = [0, self.board_width // 2 - len(self.current_piece[0]) // 2]

    def get_state(self):
        """Return the current board state and the current piece in a stable 4x4 grid."""
        return {
            'board': np.copy(self.board),
            'current_piece': self.get_piece_in_stable_grid(),
            'current_position' : self.current_position
        }
        
    def render(self, mode='human', given_piece=None, given_position=None):
        # create screen if not existent
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.board_width * self.cell_size, self.board_height * self.cell_size))
            pygame.display.set_caption("Tetris Environment")
        
        self.screen.fill((50, 50, 50))  # Fill background with black
        
        # Draw the board
        for row in range(self.board_height):
            for col in range(self.board_width):
                color = (250, 0, 0) if self.board[row, col] else (10, 10, 10)
                pygame.draw.rect(self.screen, color,
                               pygame.Rect(col * self.cell_size, row * self.cell_size, 
                                         self.cell_size - 1, self.cell_size - 1))
        
        # Draw current piece
        for row, line in enumerate(self.current_piece):
            for col, cell in enumerate(line):
                if cell:
                    x, y = self.current_position[0] + row, self.current_position[1] + col
                    if 0 <= x < self.board_height and 0 <= y < self.board_width:
                        pygame.draw.rect(self.screen, self.current_piece_color,
                                       pygame.Rect(y * self.cell_size, x * self.cell_size, 
                                                 self.cell_size - 1, self.cell_size - 1))
                        
        # Draw the given piece at the current position if provided
        if given_piece is not None and given_position is not None:
            for row, line in enumerate(given_piece):
                for col, cell in enumerate(line):
                    if cell:
                        x = given_position[0] + row
                        y = given_position[1] + col
                        if 0 <= x < self.board_height and 0 <= y < self.board_width:
                            pygame.draw.rect(self.screen, (0, 0, 100),
                                            pygame.Rect(y * self.cell_size, x * self.cell_size, 
                                                        self.cell_size - 1, self.cell_size - 1))
        font = pygame.font.Font(None, 36)
        text = font.render(f"Lines Cleared: {self.lines_cleared_count}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))
            
        pygame.display.flip()

        if self.time_delay != 0:
            time.sleep(self.time_delay)
    
    def move_left(self):
        self.current_position[1] -= 1
        if not self.is_valid_position():
            self.current_position[1] += 1
    
    def move_right(self):
        self.current_position[1] += 1
        if not self.is_valid_position():
            self.current_position[1] -= 1
    
    def rotate(self):
        self.current_piece = np.rot90(self.current_piece, -1)
        if not self.is_valid_position():
            self.current_piece = np.rot90(self.current_piece)

    def is_valid_position(self):
        for row, line in enumerate(self.current_piece):
            for col, cell in enumerate(line):
                if cell:
                    x, y = self.current_position[0] + row, self.current_position[1] + col
                    if x >= self.board_height or y < 0 or y >= self.board_width or (x >= 0 and self.board[x, y]):
                        return False
        return True