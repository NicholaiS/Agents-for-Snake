import random
import numpy as np
from snake_game import Game

class QLAgent:
    def __init__(self, action_space, state_space, learning_rate=0.01, discount=0.99, exploration_rate=1.0, min_exploration=0.001, exploration_decay=0.9995):
        self.action_space = action_space
        self.state_space = state_space
        self.learning_rate = learning_rate
        self.discount = discount
        self.exploration_rate = exploration_rate
        self.min_exploration = min_exploration
        self.exploration_decay = exploration_decay
        self.q_table = np.zeros((state_space, action_space))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, self.action_space - 1)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount * next_max)
        self.q_table[state, action] = new_value

    def decay_exploration(self):
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)

def encode_state(game):
    head_x, head_y = game.snake.body[0]
    prev_x, prev_y = game.snake.body[1] if len(game.snake.body) > 1 else (head_x, head_y)

    moving_right = int(head_x > prev_x)
    moving_left = int(head_x < prev_x)
    moving_up = int(head_y < prev_y)
    moving_down = int(head_y > prev_y)

    food_directions = [
        int(head_x < game.food[0]),
        int(head_x > game.food[0]),
        int(head_y < game.food[1]),
        int(head_y > game.food[1])
    ]

    danger = [
        int((head_x - 1, head_y) in game.snake.body or head_x - 1 < 0),
        int((head_x + 1, head_y) in game.snake.body or head_x + 1 >= game.grid_size),
        int((head_x, head_y - 1) in game.snake.body or head_y - 1 < 0),
        int((head_x, head_y + 1) in game.snake.body or head_y + 1 >= game.grid_size)
    ]

    state_index = 0
    for index, value in enumerate(food_directions + danger + [moving_right, moving_left, moving_up, moving_down]):
        state_index |= (value << index)

    return state_index

def calculate_reward(game, prev_score):
    if game.snake.is_collision():
        return -100

    head_x, head_y = game.snake.get_head_position()
    body_without_head = game.snake.body[1:] if len(game.snake.body) > 1 else []

    directions = [
        (head_x, head_y - 1),
        (head_x, head_y + 1),
        (head_x - 1, head_y),
        (head_x + 1, head_y) 
    ]
    escape_routes = 0
    for dx, dy in directions:
        if (dx, dy) not in body_without_head and 0 <= dx < game.grid_size and 0 <= dy < game.grid_size:
            escape_routes += 1

    if escape_routes == 0:
        return -50
    
    if game.score > prev_score:
        return 15

    return 0

def train_agent(episodes=100000):
    game = Game(grid_size=5)
    agent = QLAgent(action_space=4, state_space=2**12)
    scores = []
    rewards = []
    
    avg_scores_file = open('Q-tables/log_avg_scores.txt', 'w')
    avg_rewards_file = open('Q-tables/log_avg_rewards.txt', 'w')

    print("Starting training...")
    for episode in range(episodes):
        game.reset_game()
        state = encode_state(game)
        episode_rewards = 0

        while not game.game_over:
            action = agent.choose_action(state)
            prev_score = game.score
            game.update(action)
            reward = calculate_reward(game, prev_score)
            episode_rewards += reward
            next_state = encode_state(game)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            if game.won:
                break
        
        scores.append(game.score)
        rewards.append(episode_rewards)
        agent.decay_exploration()

        if episode % 100 == 0:
            avg_reward_per_episode = np.mean(rewards[-100:])
            avg_score_per_episode = np.mean(scores[-100:])
            print(f'Episode: {episode}, Average Score per Episode: {avg_score_per_episode}, Average Reward per Episode: {avg_reward_per_episode}, Exploration Rate: {agent.exploration_rate}')
            
            avg_scores_file.write(f'{avg_score_per_episode},')
            avg_rewards_file.write(f'{avg_reward_per_episode},')

        if episode % 1000 == 0:
            np.save(f'Q-tables/q_table_{episode}.npy', agent.q_table)
            
    np.save('Q-tables/q_table_final.npy', agent.q_table)
    
    # Close log files
    avg_scores_file.close()
    avg_rewards_file.close()

if __name__ == '__main__':
    train_agent()