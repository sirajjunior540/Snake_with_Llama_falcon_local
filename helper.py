import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import deque
from snake_game import Direction, Point, SnakeGameAI
from IPython import display

plt.ion()

# def plot(scores, mean_scores):
#     display.clear_output(wait=True)
#     display.display(plt.gcf())
#     plt.clf()
#     plt.title('Training...')
#     plt.xlabel('Number of Games')
#     plt.ylabel('Score')
#     plt.plot(scores)
#     plt.plot(mean_scores)
#     plt.ylim(ymin=0)
#     plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
#     plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
#     plt.show()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    # plt.show()
    plt.pause(1)
    # plt.ioff()  # Turn off interactive mode
    plt.close()  # Close the plot window
