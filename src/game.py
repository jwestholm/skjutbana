class Game:
    def __init__(self):
        self.running = True

    def run(self):
        while self.running:
            self.update()

    def update(self):
        # Game logic and updates go here
        pass

if __name__ == '__main__':
    game = Game()
    game.run()