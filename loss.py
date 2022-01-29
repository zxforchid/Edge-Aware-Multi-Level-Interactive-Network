#coding=utf-8
import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_loss(n):
    y = []
    enc = torch.load('loss')
    tempy = list(enc)
    y += tempy
    x = range(0,len(y))
    plt.plot(x, y, '.-')
    plt_title = 'epoch1000'
    plt.title(plt_title)
    plt.xlabel('per epoch')
    plt.ylabel('LOSS')
    plt.savefig('loss.png')
    plt.show()

if __name__ == "__main__":
    plot_loss(20)