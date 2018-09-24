import numpy as np


def eps_greedy(actions, greedy_action, epsilon):
	"""
	Picks an action according to an epsilon-greedy policy
	:param actions: actions to pick form
	:param greedy_action: the greedy action
	:param epsilon: probability of picking a random action
	:return: epsilon-greedy action
	"""
	return np.random.choice((greedy_action, np.random.choice(actions)), p=[1-epsilon, epsilon])


def run_episode(env, actions, s_a_values=None):
	"""
	Run an episode given state-action value function (lookup table).
	:param env: environment to use
	:param actions: action space
	:param s_a_values: state-action value function (lookup table)
	:return: total reward from this episode.
	"""
	total_reward = 0
	done = False
	random = s_a_values is None
	observation = env.reset()

	while not done:
		if random:
			# pick a random action
			action = np.random.choice(actions)
		else:
			# pick a greedy action
			action = s_a_values.get_greedy_action(observation)

		# take a step
		observation, reward, done = env.step(action)
		total_reward += reward

		if done:
			break

	return total_reward
