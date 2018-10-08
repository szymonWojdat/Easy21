from classes.Easy21 import Easy21
from classes.LookupTable import LookupTableGeneric
from functions.monte_carlo import learn_mc_episode
from functions.sarsa import learn_sarsa_episode_table, learn_sarsa_episode_lfa
from functions.common import run_episode, run_episode_lfa, phi_fn
from tqdm import tqdm
import pickle
import numpy as np


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
	for _ in tqdm(range(num_learn_episodes), desc='Learning MC and Sarsa'):
		mc_value_table, mc_reps_table = learn_mc_episode(env1, action_space, mc_value_table, mc_reps_table, n_zero)
		sarsa_value_table, sarsa_reps_table = learn_sarsa_episode_table(
			env1, state_space, action_space, sarsa_value_table, sarsa_reps_table, n_zero, lambda_val)

	# playing
	for _ in tqdm(range(num_run_episodes), desc='Playing MC, Sarsa and random'):
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


def learn_sarsa_lfe(num_learn_episodes, num_run_episodes, lambda_val):
	"""
	Performs Monte-Carlo and Sarsa(lambda) learning and evaluation
	:param num_learn_episodes: number of learning episodes
	:param num_run_episodes:  number of testing episodes
	:param lambda_val: lambda constant for Sarsa(lambda)
	:return: -
	"""
	env1 = Easy21()
	state_space = env1.get_state_space()
	action_space = env1.get_action_space()

	weights_shape = (36, 1)
	weights = np.zeros(weights_shape)

	sarsa_total = 0
	random_total = 0

	# learning
	for _ in tqdm(range(num_learn_episodes), desc='Learning Sarsa LFE'):
		weights = learn_sarsa_episode_lfa(
			env1, state_space, action_space, phi_fn, weights, 0.01, 0.05, lambda_value=lambda_val)

	# playing
	for _ in tqdm(range(num_run_episodes), desc='Playing Sarsa LFE and random'):
		sarsa_total += run_episode_lfa(env1, action_space, weights)
		random_total += run_episode(env1, action_space)

	print('\nExpected reward in Sarsa = {}'.format(sarsa_total / num_run_episodes))
	print('\nExpected reward in random = {}'.format(random_total / num_run_episodes))


def main():
	# parameters
	n_learn_ep = 10 ** 4
	n_run_ep = 10 ** 5
	n0 = 100
	lmbd = 1.0

	# learn_mc_and_sarsa(n_learn_ep, n_run_ep, n0, lmbd)
	learn_sarsa_lfe(n_learn_ep, n_run_ep, lmbd)


if __name__ == '__main__':
	main()
