from tetris_env import TetrisEnv
from tetris_agent import TetrisAgent

if __name__ == "__main__":
    full_games = 100

    for i in range(full_games):
        env = TetrisEnv(time_delay=0.01)
        agent = TetrisAgent(env, is_debug=False)
        done = False
        state = env.reset()
        while not done:
            env.render()
            if env.new_piece_spawned:
                agent.execute_best_move(render=True)
                env.new_piece_spawned = False
            state, done = env.step(-1)
        print(f'Game: ({i} / {full_games}) | Score: {env.lines_cleared_count}')
    
    env.close()