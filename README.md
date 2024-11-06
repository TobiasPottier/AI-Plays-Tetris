# AI-Plays-Tetris

This project is about an AI-like agent playing Tetris, employing a rule-based approach rather than traditional AI techniques. The agent analyzes all possible positions and rotations for each piece, selecting the best one based on a weighted reward system. Weights were optimized using simulated annealing.

![vid1](https://github.com/user-attachments/assets/3bef707d-8d46-4c92-9821-7696f0ab84a9)

**How It Works**
---------------
The agent evaluates each move and chooses the placement with the highest reward, considering factors like:

- Reducing holes
- Keeping the piece low on the board
- Clearing lines
- Amount of empty columns (pillars)


![vid2](https://github.com/user-attachments/assets/65775d46-c1b0-4e00-9d42-385d41a76a81)

**Instalation**
---------------

1. Clone the repository:
```
git clone https://github.com/username/AI-Plays-Tetris.git
```

2. Install dependencies and run:
```
# Install required packages
pip install -r requirements.txt
```
```
# Run Tetris agent
python main.py
```
