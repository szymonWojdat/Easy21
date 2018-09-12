from Easy21 import Easy21
from LookupTable import LookupTable
import numpy as np
import pickle


def eps_greedy(actions, greedy_action, epsilon):
	return np.random.choice((greedy_action, np.random.choice(actions)), p=[1-epsilon, epsilon])


def learn_mc_episode(env, actions, s_a_values, s_a_reps, n0):
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


def learn_sarsa_episode(env, actions, s_a_values, s_a_et, n0, lambda_value):
	"""
	Executes one episode of Sarsa(lambda) learning
	:param env: environment to use
	:param actions: action space of env
	:param s_a_values: state-action value function (lookup table)
	:param s_a_et: state-action eligibility traces (lookup table)
	:param n0: influences epsilon in epsilon-greedy policy
	:param lambda_value: lambda parameter value for Sarsa(lambda)
	:return: updated state-action value function and eligibility traces
	"""
	total_reward = 0
	state_memory = []
	observation = env.reset()
	player, dealer = observation
	for _ in range(1000):
		# pick an episilon-greedy action
		greedy_action = s_a_values.get_greedy_action(player, dealer)
		ns = s_a_et.get(player, dealer, 'hit') + s_a_et.get(player, dealer, 'stick')  # TODO - check if s_a_reps isn't necessary here
		epsilon = n0 / (n0 + ns)
		action = eps_greedy(actions, greedy_action, epsilon)
		s_a_et.increment(player, dealer, action)  # update N
		state_memory.append(((player, dealer), action))

		observation, reward, done = env.step(action)
		total_reward += reward

		# TODO - change this, it's basically copied from MC
		for observation, action in state_memory:
			player, dealer = observation
			alpha = 1 / s_a_et.get(player, dealer, action)
			delta_q = alpha * (total_reward - s_a_values.get(player, dealer, action))
			s_a_values.update(player, dealer, action, delta_q)  # update Q

		if done:
			break
	else:
		msg = 'Something went wrong and the loop did not break, most recent observation: {}'.format(observation)
		raise RuntimeWarning(msg)
	return s_a_values, s_a_et


def run_episode(env, actions, s_a_values=None):
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
