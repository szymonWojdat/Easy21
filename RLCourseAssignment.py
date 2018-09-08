from Easy21 import Easy21
from LookupTable import LookupTable
import numpy as np


def eps_greedy(actions, greedy_action, epsilon):
	return np.random.choice(greedy_action, np.random.choice(actions), p=[1-epsilon, epsilon])


def run_episode(env):
	total_reward = 0
	actions = env.get_action_space()
	table = LookupTable()
	observation = env.reset()
	for _ in range(1000):
		action = np.random.choice(actions)  # choosing a random action
		# pick an episilon-greedy action
		greedy_action = table.get_greedy_action(observation)
		epsilon = 1  # TODO - implement this correctly
		action = eps_greedy(actions, greedy_action, epsilon)
		observation, reward, done = env.step(action)
		total_reward += reward
		# update N
		if done:
			# update Q
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
