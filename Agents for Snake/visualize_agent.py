import pygame
import numpy as np
from snake_game import Game
from ql_training import encode_state
from ga_binary_training import decode_genome

class Visualizer:
    def __init__(self, game, agent, cell_size=30, frame_rate=15, agent_type='QL'):
        self.game = game
        self.agent = agent
        self.cell_size = cell_size
        self.frame_rate = frame_rate
        self.agent_type = agent_type
        pygame.init()
        self.screen = pygame.display.set_mode((game.grid_size * self.cell_size, game.grid_size * self.cell_size))
        pygame.display.set_caption('Snake Game')
        self.font = pygame.font.Font(None, 42)

    def draw_grid(self):
        for x in range(0, self.game.grid_size * self.cell_size, self.cell_size):
            for y in range(0, self.game.grid_size * self.cell_size, self.cell_size):
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)

    def draw_snake(self):
        for part in self.game.snake.body:
            rect = pygame.Rect(part[0] * self.cell_size, part[1] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (0, 128, 0), rect)

    def draw_food(self):
        food = self.game.food
        rect = pygame.Rect(food[0] * self.cell_size, food[1] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (255, 0, 0), rect)

    def draw_score(self):
        score_surface = self.font.render(f'Score: {self.game.score}', True, (255, 0, 0))
        self.screen.blit(score_surface, (10, 10))

    def update_frame(self):
        self.screen.fill((0, 0, 0))
        self.draw_grid()
        self.draw_snake()
        self.draw_food()
        self.draw_score()
        pygame.display.flip()

    def play_game(self):
        clock = pygame.time.Clock()
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            if self.agent_type == 'QL':
                state = encode_state(self.game)
                action = np.argmax(self.agent[state])
            elif self.agent_type == 'GA':
                action = decode_genome(self.agent, self.game)

            self.game.update(action)
            self.update_frame()

            if self.game.snake.is_collision():
                print("Collision detected, game over.")
                done = True

            if self.game.won:
                print("Victory! The snake has filled the grid.")
                done = True

            clock.tick(self.frame_rate)

        pygame.quit()

def main():
    try:
        game = Game(grid_size=20)
        ql_agent = np.load('Q-tables/q_table_final.npy')
        ga_binary_agent = np.load('Best individuals binary/best_fitness_genome.npy')
        
        ql_visualizer = Visualizer(game, ql_agent, 30, 15, agent_type='QL')
        ga_binary_visualizer = Visualizer(game, ga_binary_agent, 30, 15, agent_type='GA')
        
        agent_type = 'GA'
        if agent_type.upper() == 'QL':
            ql_visualizer.play_game()
        elif agent_type.upper() == 'GA':
            ga_binary_visualizer.play_game()

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == '__main__':
    main()
