import random

class Snake:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        mid_point = self.grid_size // 2
        self.body = [(mid_point, mid_point), (mid_point, mid_point + 1)]
        self.direction = 'UP'
        self.growing = False

    def move(self):
        head_x, head_y = self.get_head_position()
        direction_moves = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}
        move_x, move_y = direction_moves[self.direction]
        new_head = (head_x + move_x, head_y + move_y)
        self.body.insert(0, new_head)
        if not self.growing:
            self.body.pop()
        self.growing = False

    def grow(self):
        self.growing = True

    def change_direction(self, new_direction):
        opposite_directions = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
        if new_direction != opposite_directions[self.direction]:
            self.direction = new_direction

    def get_head_position(self):
        return self.body[0]

    def is_collision(self):
        head = self.get_head_position()
        if head[0] < 0 or head[0] >= self.grid_size or head[1] < 0 or head[1] >= self.grid_size:
            return True
        if head in self.body[1:-1]:
            return True
        return False

class Game:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.snake = Snake(grid_size)
        self.food = None
        self.score = 0
        self.game_over = False
        self.won = False
        self.spawn_food()

    def spawn_food(self):
        while True:
            position = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if position not in self.snake.body:
                self.food = position
                break

    def update(self, action):
        if self.game_over or self.won:
            return

        self.snake.change_direction(['UP', 'DOWN', 'LEFT', 'RIGHT'][action])
        self.snake.move()

        if self.snake.get_head_position() == self.food:
            self.snake.grow()
            self.score += 1
            if len(self.snake.body) == self.grid_size * self.grid_size:
                self.won = True
                print("Win detected")
            else:
                self.spawn_food()

        if self.snake.is_collision():
            self.game_over = True

    def reset_game(self):
        self.snake.reset()
        self.score = 0
        self.game_over = False
        self.won = False
        self.spawn_food()
