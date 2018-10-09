# Easy21
This repository contains solutions to the [programming assignment](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf) from [David Silver's reinforcement learning course](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ).

The main task was to implement two reinforcement learning algorithms - Monte-Carlo Control and Sarsa(λ) (a verson of temporal difference learning, also known as TD learning) - and an environment - Easy21 - a simplified version of the card game Black Jack (or 21) and then apply those algorithms to solving the environment.

All functions and classes in this project were designed with the intention of being as generic as possible. Full compatibility with OpenAI Gym - WIP.

Presented below solutions have been implemented using Python 3. All solutions can be executed one after another by running:

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
The graph below presents changes of mean-squared error over time for two values of lambda: 0 and 1. A data point was captured each 1000 episodes, over the course of 100,000 episodes.
![](https://raw.githubusercontent.com/szymonWojdat/Easy21/master/graphs/sarsa_mse_over_time_lfa.png)

The graph below presents changes of mean-squared error for different values of lambda: 0, 0.1, 0.2, ..., 1. For each value, 1000 episodes have been evaluated.

![](https://raw.githubusercontent.com/szymonWojdat/Easy21/master/graphs/sarsa_mse_over_lambda_lfa.png)

## Discussion
* What are the pros and cons of bootstrapping in Easy21?

  Pros:
    * Generally speeds up the learning process - more updates per episode.
  
  Cons: 
    * The rewards happen only at the end of an episode, so there is a chance we can lower the value of a "good" state if it's              not terminal (as R = 0 for all non-terminal states).
    * Increases the variance, as the environment isn't deterministic - sometimes we might get "unlucky" and the error (over- or undervalued state-action pair) keeps propagating on other state-action pairs (through R + Q(s', a')).

* Would you expect bootstrapping to help more in blackjack or Easy21? Why?
  In blackjack as there are effectively more states because we keep playing with the same pile of cards. So in case we don't apply bootstrapping, then we lose the information for all those non-terminal states - we'll never know what was the "state of the pile" two hits ago but it's important as, for example, if we're running low on 10s, then the odds of getting a 10 are low in that state, so hitting while sitting at 12 should be valued (at least slightly) higher than normally.

* What are the pros and cons of function approximation in Easy21?

  Pros:
    * Takes less time to train as state-action values might be (and probably are) at least "partially" linearly dependent, so there is no need to run into every single state-action combination numerous times.
  
  Cons: 
    * The value function might not be linear or in general - more complex than the approximator.
    * In table lookups we start at 0, which is neutral but this isn't always true in case of function approximation - after a few updates, we might receive a non-zero value for a previously unseen state-action pair.

* How would you modify the function approximator suggested in this section to get better results in Easy21?
  
  I would try a neural network: Input: player sum, dealer card (state). Output: hit value, stick value. Make it stochastic, eg. P(hit) = Q(hit)/(Q(hit) + Q(stick)). Experiment with different numbers of hidden layers and hidden units. Maybe even try two separate neural nets - one for hit and one for stick.
