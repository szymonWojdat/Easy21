from Easy21 import Easy21
from LookupTable import LookupTable
import numpy as np


def eps_greedy(actions, greedy_action, epsilon):
	return np.random.choice((greedy_action, np.random.choice(actions)), p=[1-epsilon, epsilon])


def run_episode(env, table, n0):
	total_reward = 0
	state_memory = []
	actions = env.get_action_space()
	observation = env.reset()
	player, dealer = observation
	for _ in range(1000):
		# pick a random action
		# action = np.random.choice(actions)

		# pick an episilon-greedy action
		greedy_action = table.get_greedy_action(player, dealer)
		ns = table.get_n(player, dealer, 'hit') + table.get_n(player, dealer, 'stick')
		epsilon = n0/(n0 + ns)
		action = eps_greedy(actions, greedy_action, epsilon)
		table.inc_n(player, dealer, action)  # update N
		state_memory.append(((player, dealer), action))

		observation, reward, done = env.step(action)
		total_reward += reward

		if done:  # i think we shouldn't put the update in the terminal state, as there's no action there
			for observation, action in state_memory:
				player, dealer = observation
				alpha = 1/table.get_n(player, dealer, action)
				delta_q = alpha * (total_reward - table.get_q(player, dealer, action))
				table.update_q(player, dealer, action, delta_q)  # update Q
			break
	else:
		msg = 'Something went wrong and the loop did not break, most recent observation: {}'.format(observation)
		raise RuntimeWarning(msg)
	return total_reward, table


if __name__ == '__main__':
	env1 = Easy21()
	total = 0
	num_episodes = 10**4
	t = LookupTable()
	n_zero = 100  # this should be parametrized later
	for i in range(num_episodes):
		if i % 1000 == 0:
			print(i)
		r, t = run_episode(env1, t, n_zero)
		total += r
	# TODO - save the table (t) here or sth, we probably wanna see its final score (r), not total
	# print('\nExpected reward = {}'.format(total/num_episodes))
	t.pprint()
