import numpy as np
import os
from snake_game import Game
from tqdm import tqdm

def initialize_population(pop_size, genome_length):
    """Initialize a population of random genomes"""
    return [np.random.randint(0, 2, size=genome_length) for _ in range(pop_size)]

def select_movement(game):
    head_x, head_y = game.snake.get_head_position()
    food_x, food_y = game.food
    current_direction = game.snake.direction

    dangers = {
        'left': (head_x - 1, head_y) in game.snake.body or head_x - 1 < 0,
        'right': (head_x + 1, head_y) in game.snake.body or head_x + 1 >= game.grid_size,
        'up': (head_x, head_y - 1) in game.snake.body or head_y - 1 < 0,
        'down': (head_x, head_y + 1) in game.snake.body or head_y + 1 >= game.grid_size
    }

    preferences = {
        'left': (head_x > food_x) * (current_direction != 'right'),
        'right': (head_x < food_x) * (current_direction != 'left'),
        'up': (head_y > food_y) * (current_direction != 'down'),
        'down': (head_y < food_y) * (current_direction != 'up')
    }

    scores = {direction: preferences[direction] * (not dangers[direction])
              for direction in ['left', 'right', 'up', 'down']}

    best_direction = max(scores, key=scores.get)
    action_map = {'left': 2, 'right': 3, 'up': 0, 'down': 1}

    return action_map[best_direction]

def fitness_function(genome, game, episodes=100, switch_generation=250, current_generation=0):
    """Calculate fitness score and return both fitness and game score."""
    total_fitness = 0
    scores = []

    for _ in range(episodes):
        game.reset_game()
        frames_since_last_fruit = 0
        frames_alive = 0

        while not game.game_over and not game.won:
            action = select_movement(game)
            game.update(action)
            frames_alive += 1
            frames_since_last_fruit += 1

            if frames_since_last_fruit >= 100000:
                game.game_over = True
                frames_alive -= 50

            if game.snake.get_head_position() == game.food:
                frames_since_last_fruit = 0

        score = len(game.snake.body) - 1
        scores.append(score)
        if current_generation < switch_generation:
            fitness = (score ** 3) * frames_alive
        else:
            fitness = ((score * 2) ** 2) * (frames_alive ** 1.5)

        total_fitness += fitness
        avg_game_score = np.mean(scores[-episodes:])

    average_fitness = total_fitness / episodes
    return average_fitness, avg_game_score

def crossover(parent1, parent2):
    """Perform crossover between two genomes"""
    cross_point = np.random.randint(len(parent1))
    child1 = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
    child2 = np.concatenate([parent2[:cross_point], parent1[cross_point:]])
    return child1, child2

def mutate(genome, mutation_rate=0.01):
    """Mutate a genome by flipping bits with a given mutation rate"""
    for i in range(len(genome)):
        if np.random.rand() < mutation_rate:
            genome[i] = 1 - genome[i]
    return genome

def roulette_wheel_selection(population, fitness_scores):
    """Select an individual from the population using roulette wheel selection"""
    total_fitness = sum(fitness_scores)
    pick = np.random.uniform(0, total_fitness)
    current = 0
    for individual, score in zip(population, fitness_scores):
        current += score
        if current > pick:
            return individual

def train_ga(population_size=50, genome_length=200, generations=1000):
    population = initialize_population(population_size, genome_length)
    game = Game(grid_size=20)

    directory = 'Best individuals binary'
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    # Open log files
    fitness_scores_file = open('Best individuals binary/fitness_scores.txt', 'w')
    game_scores_file = open('Best individuals binary/game_scores.txt', 'w')

    for generation in range(generations):
        fitness_scores, avg_game_scores = [], []
        for genome in tqdm(population, desc=f"Assessing Generation {generation+1}"):
            fitness, avg_game_score = fitness_function(genome, game, current_generation=generation)
            fitness_scores.append(fitness)
            avg_game_scores.append(avg_game_score)

        best_index = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_index]
        avg_fitness = np.mean(fitness_scores[-population_size:])
        avg_game_score = np.mean(avg_game_scores[-population_size:])
        best_genome = population[best_index]
        
        fitness_scores_file.write(f'{avg_fitness},')
        game_scores_file.write(f'{avg_game_score},')

        if (generation + 1) % 100 == 0:
            np.save(os.path.join(directory, f'best_genome_gen_{generation+1}.npy'), best_genome)
            print(f"Saved best genome of Generation {generation + 1} with score: {best_fitness}")

        print(f"Generation {generation + 1}: Average Fitness Score per Generation = {avg_fitness}, Average Game Score per Generation = {avg_game_score}")

        new_population = []
        for _ in range(population_size // 2):
            parent1 = roulette_wheel_selection(population, fitness_scores)
            parent2 = roulette_wheel_selection(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        population = new_population

    fitness_scores_file.close()
    game_scores_file.close()

if __name__ == '__main__':
    train_ga()