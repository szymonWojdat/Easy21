# Easy21
This repository contains solutions to the [programming assignment](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf) from [David Silver's reinforcement learning course](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ).

The main task was to implement two reinforcement learning algorithms - Monte-Carlo Control and Sarsa(λ) (a verson of temporal difference learning, also known as TD learning) - and an environment - Easy21 - a simplified version of the card game Black Jack (or 21) and then apply those algorithms to solving the environment.

All presented below solutions have been implemented using Python 3. All solutions can be executed one after another by running:

`python3 solutions.py`

## Monte-Carlo Control in Easy21
Monte-Carlo value function is presented below. One million of episodes has been evaluated.

![](https://raw.githubusercontent.com/szymonWojdat/Easy21/master/graphs/mc_value_function.png)

## TD Learning in Easy21
The graph below presents changes of mean-squared error over time for two values of lambda: 0 and 1. A data point was captured each 1000 episodes, over the course of one million episodes.

![](https://raw.githubusercontent.com/szymonWojdat/Easy21/master/graphs/sarsa_mse_over_time.png)

The graph below presents changes of mean-squared error for different values of lambda: 0, 0.1, 0.2, ..., 1. For each value, 1000 episodes have been evaluated.

![](https://raw.githubusercontent.com/szymonWojdat/Easy21/master/graphs/sarsa_mse_over_lambda.png)

## Linear Function Approximation in Easy21
TBD

## Discussion
TBD
