#!/usr/bin/env python3
"""Collect human demonstration data from set of games.

Usage:
python3 DeepRL/tools/get_demo.py --multi-game --num-episodes=1
    --demo-time-limit=20 --hz=60.0 --create-movie
"""
import gym
import numpy as np
import pygame

from getdemo import get_demo


def multi_game(args):
    pygame.init()
    pygame.display.set_caption('Atari 2600')

    black = (0, 0, 0)
    white = (255, 255, 255)
    games = ['Asteroids', 'BattleZone', 'Breakout', 'Hero', 'MsPacman', 'Enduro']

    display_width = 300 * len(games)
    display_height = 450

    gameDisplay = pygame.display.set_mode((display_width, display_height))

    clock = pygame.time.Clock()
    game_imgs = []
    for game in games:
        env = gym.make("{}NoFrameskip-v4".format(game))
        env.reset()
        for i in range(30):
            env.step(1)
        image = env.unwrapped.ale.getScreenRGB()
        del env

        image = np.fliplr(image)
        image = np.rot90(image)
        picture = pygame.surfarray.make_surface(image)
        picture = pygame.transform.scale(picture, (300, 420))
        game_imgs.append(picture)

    def put_image(img, x, y):
        gameDisplay.blit(img, (x, y))

    def text_objects(text, font):
        textSurface = font.render(text, True, black)
        return textSurface, textSurface.get_rect()

    def draw_start(num=0):
        pygame.draw.rect(gameDisplay, (255, 255, 255), [300*num, 350, 300, 55])
        largeText = pygame.font.Font('freesansbold.ttf', 40)
        TextSurf, TextRect = text_objects("START GAME", largeText)
        loc = [300, 900, 1500, 2100, 2700, 3300]
        TextRect.center = (loc[num] / 2, 380)
        gameDisplay.blit(TextSurf, TextRect)

    def draw_name(num=0):
        largeText = pygame.font.Font('freesansbold.ttf', 20)
        TextSurf, TextRect = text_objects(games[num], largeText)
        loc = [300, 900, 1500, 2100, 2700, 3300]
        TextRect.center = (loc[num] / 2, 435)
        gameDisplay.blit(TextSurf, TextRect)

    x = (display_width * 0)
    y = (display_height * 0)

    game_num = 0
    quit = False
    while not quit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit = True

        gameDisplay.fill(white)
        for i, game in enumerate(game_imgs):
            put_image(game, x+(300*i), y)
            draw_name(i)
            draw_start(game_num)
            pygame.draw.rect(gameDisplay, white, pygame.Rect(0, 0, (300*(1+i)), 420), 5)

        if pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            hat = joystick.get_hat(0)
            start = joystick.get_button(0)
            if hat[0] == 1:
                game_num += 1
                game_num = 0 if game_num == len(games) else game_num
            elif hat[0] == -1:
                game_num -= 1
                game_num = len(games)-1 if game_num < 0 else game_num
            if start:
                get_demo(args, game=games[game_num], pause_onstart=False)
                pygame.init()
                pygame.display.set_caption('Atari 2600')

        pygame.display.update()
        # clock.tick(60)
        clock.tick(8)

    pygame.quit()
