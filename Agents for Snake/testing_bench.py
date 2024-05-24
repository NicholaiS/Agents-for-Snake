import numpy as np
from snake_game import Game
from ql_training import encode_state
from ga_binary_training import decode_genome
from tqdm import tqdm

def load_agent(agent_type, filepath):
    if agent_type in ['ql', 'binary']:
        return np.load(filepath)
    else:
        raise ValueError("Unknown agent type")

def test_agent(grid, agent, agent_type, episodes=100000):
    game = Game(grid_size=grid)
    total_score = 0
    wins = 0

    for _ in tqdm(range(episodes), desc="Testing Agent"):
        game.reset_game()
        while not game.game_over:
            if agent_type == 'ql':
                state = encode_state(game)
                action = np.argmax(agent[state])
            elif agent_type == 'binary':
                action = decode_genome(agent, game)

            game.update(action)

            if game.won:
                wins += 1
                break

        total_score += game.score

    average_score = total_score / episodes
    return wins, average_score

def main():
    agent_type = 'binary'  # 'ql' or 'binary'
    filename = 'Q-tables/q_table_final.npy' if agent_type == 'ql' else 'Best individuals binary/best_score_genome.npy'
    
    agent = load_agent(agent_type, filename)
    
    wins, average_score = test_agent(5, agent, agent_type)
    
    with open("test_results.txt", "w") as file:
        file.write(f"Agent Type: {agent_type}\n")
        file.write(f"Total Wins: {wins}\n")
        file.write(f"Average Score: {average_score}\n")

if __name__ == '__main__':
    main()