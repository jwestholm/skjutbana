import pygame
import sys
from config import SCREEN_WIDTH, SCREEN_HEIGHT, LOADING_SCREEN_PATH

# Initialize Pygame
pygame.init()

# Set up display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Game Window')

# Load loading screen image
loading_screen = pygame.image.load(LOADING_SCREEN_PATH)

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update game state

    # Display the loading screen
    screen.blit(loading_screen, (0, 0))

    # Update the display
    pygame.display.flip()

pygame.quit()
