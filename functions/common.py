import numpy as np
from classes.LookupTable import LookupTableGeneric


# probably a good idea to rewrite these two MSE methods as one some day...
def mse_tables(t1: LookupTableGeneric, t2: LookupTableGeneric):
	state_space, action_space = t1.get_space()
	assert (state_space, action_space) == t2.get_space(), 'State/action spaces do not match'

	square_sum = 0.0
	n = 0
	for state in state_space:
		for action in action_space:
			square_sum += (t1.get(state, action) - t2.get(state, action))**2
			n += 1
	return square_sum/n


def mse_lfa_table(t, w, phi, states, actions):
	square_sum = 0.0
	n = 0
	for state in states:
		for action in actions:
			# square_sum += (t.get(state, action) - np.matmul(phi(state, action), w)) ** 2
			square_sum += (t.get(state, action) - np.dot(phi(state, action), w).flatten()[0]) ** 2
			n += 1
	return square_sum / n


def greedy_lfa(actions, s, th, phi):
	action_values = {}
	for act in actions:
		action_values[act] = q(s, act, th, phi)
	best_actions = [k for k, v in action_values.items() if v == max(action_values.values())]
	return np.random.choice(best_actions)


# TODO - merge these two methods (w/ eps_greedy_lfa)
def eps_greedy(actions, greedy_action, epsilon):
	"""
	Picks an action according to an epsilon-greedy policy
	:param actions: actions to pick form
	:param greedy_action: the greedy action
	:param epsilon: probability of picking a random action
	:return: epsilon-greedy action
	"""
	return np.random.choice((greedy_action, np.random.choice(actions)), p=[1-epsilon, epsilon])


def eps_greedy_lfa(actions, s, th, epsilon, phi):
	greedy_action = greedy_lfa(actions, s, th, phi)
	return eps_greedy(actions, greedy_action, epsilon)


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


def run_episode_lfa(env, actions, theta):
	"""
	Run an episode given state-action value function (lookup table).
	:param env: environment to use
	:param actions: action space
	:param theta: vector of weights
	:return: total reward from this episode.
	"""
	total_reward = 0
	done = False
	observation = env.reset()

	while not done:
		# pick a greedy action
		action = greedy_lfa(actions, observation, theta, phi_fn)

		# take a step
		observation, reward, done = env.step(action)
		total_reward += reward

		if done:
			break

	return total_reward


def phi_fn(state, action):
	player, dealer = state
	phi = np.zeros(36)
	player_values = (range(1, 7), range(4, 10), range(7, 13), range(10, 16), range(13, 19), range(16, 22))
	dealer_values = (range(1, 5), range(4, 8), range(7, 11))
	action_values = (['hit'], ['stick'])  # 1-el arrays as i want to use 'in' for consistency

	count = 0
	for p in player_values:
		for d in dealer_values:
			for a in action_values:
				if player in p and dealer in d and action in a:
					phi[count] = 1
				count += 1
	return phi


def q(s, a, th, phi):
	return np.dot(phi(s, a), th).flatten()[0]
