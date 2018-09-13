from Easy21 import Easy21
from LookupTable import LookupTable
import numpy as np
import pickle


def eps_greedy(actions, greedy_action, epsilon):
	"""
	Picks an action according to an epsilon-greedy policy
	:param actions: actions to pick form
	:param greedy_action: the greedy action
	:param epsilon: probability of picking a random action
	:return: epsilon-greedy action
	"""
	return np.random.choice((greedy_action, np.random.choice(actions)), p=[1-epsilon, epsilon])


def learn_mc_episode(env, actions, s_a_values, s_a_reps, n0):
	"""
	Executes one episode of Monte-Carlo learning (control)
	:param env: environment to use
	:param actions: action space of env
	:param s_a_values: state-action value function (lookup table)
	:param s_a_reps: state-action counts
	:param n0: influences epsilon in epsilon-greedy policy
	:return: updated state-action value function and state-action counts
	"""
	total_reward = 0
	state_memory = []
	observation = env.reset()
	player, dealer = observation
	for _ in range(1000):
		# pick an episilon-greedy action
		greedy_action = s_a_values.get_greedy_action(player, dealer)
		ns = s_a_reps.get(player, dealer, 'hit') + s_a_reps.get(player, dealer, 'stick')
		epsilon = n0/(n0 + ns)
		action = eps_greedy(actions, greedy_action, epsilon)
		s_a_reps.increment(player, dealer, action)  # update N
		state_memory.append(((player, dealer), action))

		observation, reward, done = env.step(action)
		total_reward += reward

		if done:
			for observation, action in state_memory:
				player, dealer = observation
				alpha = 1/s_a_reps.get(player, dealer, action)
				delta_q = alpha * (total_reward - s_a_values.get(player, dealer, action))
				s_a_values.update(player, dealer, action, delta_q)  # update Q
			break
	else:
		msg = 'Something went wrong and the loop did not break, most recent observation: {}'.format(observation)
		raise RuntimeWarning(msg)
	return s_a_values, s_a_reps


def learn_sarsa_episode(env, actions, s_a_values, s_a_reps, n0, lambda_value, gamma):
	"""
	Executes one episode of Sarsa(lambda) learning
	:param env: environment to use
	:param actions: action space of env
	:param s_a_values: state-action value function (lookup table)
	:param s_a_reps: state-action counts
	:param n0: influences epsilon in epsilon-greedy policy
	:param lambda_value: lambda parameter value for Sarsa(lambda)
	:param gamma: discount factor
	:return: updated state-action value function, state-action counts and eligibility traces
	"""
	total_reward = 0
	state_memory = []
	observation = env.reset()
	player, dealer = observation
	s_a_et = LookupTable(actions)

	# pick an episilon-greedy action
	greedy_action = s_a_values.get_greedy_action(player, dealer)
	ns = s_a_reps.get(player, dealer, 'hit') + s_a_reps.get(player, dealer, 'stick')
	epsilon = n0 / (n0 + ns)
	action = eps_greedy(actions, greedy_action, epsilon)
	s_a_reps.increment(player, dealer, action)  # update N
	state_memory.append(((player, dealer), action))

	for _ in range(1000):
		observation, reward, done = env.step(action)
		player_prime, dealer_prime = observation
		total_reward += reward

		# pick an episilon-greedy action
		greedy_action = s_a_values.get_greedy_action(player_prime, dealer_prime)
		ns = s_a_reps.get(player_prime, dealer_prime, 'hit') + s_a_reps.get(player, dealer, 'stick')
		epsilon = n0 / (n0 + ns)
		action_prime = eps_greedy(actions, greedy_action, epsilon)
		s_a_reps.increment(player_prime, dealer_prime, action)  # update N

		delta = reward + gamma * s_a_values.get(player_prime, dealer_prime, action_prime) -\
			s_a_values.get(player, dealer, action)

		for _observation, _action in state_memory:
			_player, _dealer = _observation
			alpha = 1 / s_a_reps.get(_player, _dealer, _action)
			q_update = alpha * delta * s_a_et.get(_player, _dealer, _action)
			s_a_values.update(_player, _dealer, _action, q_update)  # update Q
			et_update = gamma * lambda_value * s_a_et.get(_player, _dealer, _action)
			s_a_et.set(_player, _dealer, _action, et_update)

		player, dealer, action = player_prime, dealer_prime, action_prime
		state_memory.append(((player, dealer), action))

		if done:
			break
	else:
		msg = 'Something went wrong and the loop did not break, most recent observation: {}'.format(observation)
		raise RuntimeWarning(msg)
	return s_a_values, s_a_reps


def run_episode(env, actions, s_a_values=None):
	"""
	Run an episode given state-action value function (lookup table).
	:param env: environment to use
	:param actions: action space
	:param s_a_values: state-action value function (lookup table)
	:return: total reward from this episode.
	"""
	total_reward = 0
	observation = env.reset()
	player, dealer = observation
	random = s_a_values is None
	for _ in range(1000):
		if random:
			# pick a random action
			action = np.random.choice(actions)
		else:
			# pick a greedy action
			action = s_a_values.get_greedy_action(player, dealer)

		# take a step
		observation, reward, done = env.step(action)
		total_reward += reward

		if done:
			break
	else:
		msg = 'Something went wrong and the loop did not break, most recent observation: {}'.format(observation)
		raise RuntimeWarning(msg)
	return total_reward


if __name__ == '__main__':
	# parameters
	num_learn_episodes = 10 ** 5
	num_run_episodes = 10 ** 4
	n_zero = 100

	env1 = Easy21()
	action_space = env1.get_action_space()
	value_table = LookupTable(action_space)
	reps_table = LookupTable(action_space)
	total = 0
	total_random = 0

	# learning
	for i in range(num_learn_episodes):
		if i % 10000 == 0:
			print(i)
		value_table, reps_table = learn_mc_episode(env1, action_space, value_table, reps_table, n_zero)

	# playing
	for _ in range(num_run_episodes):
		total += run_episode(env1, action_space, value_table)

	for _ in range(num_run_episodes):
		total_random += run_episode(env1, action_space)

	print('\nExpected reward in MC = {}'.format(total/num_run_episodes))
	print('\nExpected reward in random = {}'.format(total_random/num_run_episodes))

	file = open('dumps/monte_carlo_table.pkl', 'wb')
	pickle.dump(value_table, file)
	# to open, use: pickle.load(open('monte_carlo_table.pkl', 'rb'), encoding='UTF-8') or sys.stdout.encoding
