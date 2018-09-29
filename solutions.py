from classes.Easy21 import Easy21
from classes.LookupTable import LookupTableGeneric
from functions.monte_carlo import learn_mc_episode
from functions.sarsa import learn_sarsa_episode
from functions.common import mse
import functions.plotting as plotting
import numpy as np
import pickle
import os
import sys
from tqdm import tqdm


def run_monte_carlo(num_episodes, n_zero, save_plots=True):
	env1 = Easy21()
	state_space = env1.get_state_space()
	action_space = env1.get_action_space()

	mc_value_table = LookupTableGeneric(state_space, action_space)
	mc_reps_table = LookupTableGeneric(state_space, action_space)

	# learning
	for _ in tqdm(range(num_episodes), desc='Monte-Carlo Learning'):
		mc_value_table, mc_reps_table = learn_mc_episode(env1, action_space, mc_value_table, mc_reps_table, n_zero)

	# plotting
	plotting.plot_value_for_player_dealer(mc_value_table, action_space, save_plots)

	# saving to pickle file
	directory = 'dumps'
	if not os.path.exists(directory):
		os.makedirs(directory)
	with open(directory + '/mc_table.pkl', 'wb') as file:
		pickle.dump(mc_value_table, file)


def run_sarsa(num_episodes, n_zero, mc_value_table, save_plots=True):
	assert num_episodes > 1000, 'Number of episodes must be greater than 1000'
	env1 = Easy21()
	state_space = env1.get_state_space()
	action_space = env1.get_action_space()

	# MSE over time for lambda = 0 and lambda = 1

	sarsa_value_table_0 = LookupTableGeneric(state_space, action_space)
	sarsa_reps_table_0 = LookupTableGeneric(state_space, action_space)
	sarsa_value_table_1 = LookupTableGeneric(state_space, action_space)
	sarsa_reps_table_1 = LookupTableGeneric(state_space, action_space)

	time_steps = []
	mse_memo_0 = []
	mse_memo_1 = []

	for i in tqdm(range(num_episodes), desc='Sarsa(lambda) Learning, part 1/2'):
		if i % 1000 == 0 and not i == 0:
			time_steps.append(i)
			mse_memo_0.append(mse(mc_value_table, sarsa_value_table_0))
			mse_memo_1.append(mse(mc_value_table, sarsa_value_table_1))

		sarsa_value_table_0, sarsa_reps_table_0 = learn_sarsa_episode(
			env1, state_space, action_space, sarsa_value_table_0, sarsa_reps_table_0, n_zero, lambda_value=0)

		sarsa_value_table_1, sarsa_reps_table_1 = learn_sarsa_episode(
			env1, state_space, action_space, sarsa_value_table_1, sarsa_reps_table_1, n_zero, lambda_value=1)

	# MSE over lambda

	lambdas = []
	mse_memo_lambda = []

	for lambda_val in tqdm(np.linspace(0, 1, 10), desc='Sarsa(lambda) Learning, part 2/2'):
		lambda_val = np.round(lambda_val, 1)  # for some reason these numbers aren't always exactly round
		sarsa_value_table = LookupTableGeneric(state_space, action_space)
		sarsa_reps_table = LookupTableGeneric(state_space, action_space)
		for _ in range(1000):  # hardcoded since that's the required numebr of learning episodes per lambda value
			sarsa_value_table, sarsa_reps_table = learn_sarsa_episode(
				env1, state_space, action_space, sarsa_value_table, sarsa_reps_table, n_zero, lambda_val)
		lambdas.append(lambda_val)
		mse_memo_lambda.append(mse(mc_value_table, sarsa_value_table))

	# plotting
	plotting.plot_sarsa_mse_vs_nsteps(mse_memo_0, mse_memo_1, time_steps, save_plots)
	plotting.plot_sarsa_mse_vs_lambda(mse_memo_lambda, lambdas, save_plots)


def main():
	# parameters
	n_learn_ep = 10 ** 4
	n0 = 100

	save_plots_to_file = False

	# task 2 - Monte-Carlo control in Easy21
	run_monte_carlo(n_learn_ep, n0, save_plots=save_plots_to_file)

	# task 3 - TD Learning in Easy21
	mc_val_tab = pickle.load(open('dumps/mc_table.pkl', 'rb'), encoding=sys.stdout.encoding)
	run_sarsa(n_learn_ep, n0, mc_val_tab)


if __name__ == '__main__':
	main()
