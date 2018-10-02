from classes.Easy21 import Easy21
from classes.LookupTable import LookupTableGeneric
from functions.monte_carlo import learn_mc_episode
from functions.sarsa import learn_sarsa_episode_table
from functions.common import run_episode
import pickle


def learn_mc_and_sarsa(num_learn_episodes, num_run_episodes, n_zero, lambda_val, save=False):
	"""
	Performs Monte-Carlo and Sarsa(lambda) learning and evaluation
	:param num_learn_episodes: number of learning episodes
	:param num_run_episodes:  number of testing episodes
	:param n_zero: N0 constant used for epsilon in an epsilon-greedy policy
	:param lambda_val: lambda constant for Sarsa(lambda)
	:param save: if True, saves the results to .pkl files
	:return: -
	"""
	env1 = Easy21()
	state_space = env1.get_state_space()
	action_space = env1.get_action_space()

	mc_value_table = LookupTableGeneric(state_space, action_space)
	mc_reps_table = LookupTableGeneric(state_space, action_space)
	mc_total = 0

	sarsa_value_table = LookupTableGeneric(state_space, action_space)
	sarsa_reps_table = LookupTableGeneric(state_space, action_space)
	sarsa_total = 0

	random_total = 0

	# learning
	for i in range(num_learn_episodes):
		if i % 10000 == 0:
			print(i)
		mc_value_table, mc_reps_table = learn_mc_episode(env1, action_space, mc_value_table, mc_reps_table, n_zero)
		sarsa_value_table, sarsa_reps_table = learn_sarsa_episode_table(
			env1, state_space, action_space, sarsa_value_table, sarsa_reps_table, n_zero, lambda_val)

	# playing
	for _ in range(num_run_episodes):
		mc_total += run_episode(env1, action_space, mc_value_table)
		sarsa_total += run_episode(env1, action_space, sarsa_value_table)
		random_total += run_episode(env1, action_space)

	print('\nExpected reward in MC = {}'.format(mc_total / num_run_episodes))
	print('\nExpected reward in Sarsa = {}'.format(sarsa_total / num_run_episodes))
	print('\nExpected reward in random = {}'.format(random_total / num_run_episodes))

	if save:
		file = open('dumps/mc_table.pkl', 'wb')
		pickle.dump(mc_value_table, file)

		file = open('dumps/sarsa_table.pkl', 'wb')
		pickle.dump(sarsa_value_table, file)
		# to open, use: pickle.load(open('monte_carlo_table.pkl', 'rb'), encoding='UTF-8') or sys.stdout.encoding


def main():
	# parameters
	n_learn_ep = 10 ** 5
	n_run_ep = 10 ** 5
	n0 = 100
	lmbd = 1.0

	learn_mc_and_sarsa(n_learn_ep, n_run_ep, n0, lmbd)


if __name__ == '__main__':
	main()
