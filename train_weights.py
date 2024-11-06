import numpy as np
from tetris_agent import TetrisAgent
from tetris_env import TetrisEnv
import random
import math

# Define the simulated annealing process
class SimulatedAnnealingOptimizer:
    def __init__(self, agent, initial_weights, temperature=100.0, cooling_rate=0.9999):
        self.agent = agent
        self.weights = initial_weights
        self.temperature = temperature
        self.cooling_rate = cooling_rate

    def objective_function(self, weights):
        # Update the agent's weights
        self.agent.weights = weights

        # Run a game and calculate the score as the evaluation criteria
        full_games = 5
        average_score = 0
        score = 0
        for _ in range(full_games):
            done = False
            self.agent.env.reset()
            while not done:
                if env.new_piece_spawned:
                    self.agent.execute_best_move()
                    self.agent.env.new_piece_spawned = False
                state, done = env.step(-1)
            score += self.agent.env.lines_cleared_count
        average_score = score / full_games
        return average_score

    def perturb_weights(self):
        # Slightly adjust each weight randomly to explore the solution space
        noise_scale = 0.1
        return self.weights + np.random.normal(0, noise_scale, size=4)

    def optimize(self, iterations=1500):
        best_weights = self.weights
        best_score = self.objective_function(self.weights)
        
        for i in range(iterations):

            new_weights = self.perturb_weights()

            if i % int((iterations / 50)) == 0 and i != 0:
                new_weights = 2 * np.random.rand(4)
                new_weights[0] = abs(new_weights[0])
                new_weights[1] = abs(new_weights[1])
                new_weights[2] = -abs(new_weights[2])
                new_weights[3] = -abs(new_weights[3])
                self.weights = new_weights
                print('new weights')
            
            new_score = self.objective_function(new_weights)
            
            # Calculate acceptance probability
            delta_score = new_score - best_score
            acceptance_probability = math.exp(delta_score / self.temperature) if delta_score < 0 else 1
            
            # Decide whether to accept the new weights
            if random.random() < acceptance_probability:
                self.weights = new_weights
                best_weights = new_weights if new_score > best_score else best_weights
                best_score = new_score if new_score > best_score else best_score
            
            # Cool down the temperature
            self.temperature *= self.cooling_rate

            print(f"Iteration {i+1}, Best Score: {best_score}, Weights: {best_weights} || {self.weights} || {acceptance_probability}")

        return best_weights, best_score

# Usage example
if __name__ == "__main__":
    # Initialize the Tetris environment and agent
    env = TetrisEnv()
    agent = TetrisAgent(env, is_debug=False)
    
    # Initial weights for [lines_cleared, holes, height, empty_pillars_penalty]
    initial_weights = [0.19084932741423177, 0.6299980219054631, -0.0229841425115187, -0.9764034410377012]

    # Create optimizer and start optimization
    optimizer = SimulatedAnnealingOptimizer(agent, initial_weights)
    best_weights, best_score = optimizer.optimize()

    
    print("Best Weights:", best_weights)
    print("Best Score:", best_score)

    env.close()