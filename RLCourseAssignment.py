from Easy21 import Easy21
from random import choice
import numpy as np

ACTIONS = ('hit', 'stick')


def run_episode(env):
	total_reward = 0
	global ACTIONS

	observation = env.reset()
	for _ in range(1000):
		action = choice(ACTIONS)  # choosing a random action
		observation, reward, done = env.step(action)
		total_reward += reward
		if done:
			break
	else:
		msg = 'Something went wrong and the loop did not break, most recent observation: {}'.format(observation)
		raise RuntimeWarning(msg)
	return total_reward


if __name__ == '__main__':
	env1 = Easy21()
	total = 0
	num_episodes = 10**6
	for _ in range(num_episodes):
		total += run_episode(env1)
	print('\nExpected reward = {}'.format(total/num_episodes))
