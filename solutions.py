from classes.Easy21 import Easy21
from classes.LookupTable import LookupTableGeneric
from functions.monte_carlo import learn_mc_episode
from functions.sarsa import learn_sarsa_episode_table, learn_sarsa_episode_lfa
from functions.common import mse_tables, mse_lfa_table, phi_fn
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


def run_sarsa_table(num_episodes, n_zero, mc_value_table, save_plots=True):
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
			mse_memo_0.append(mse_tables(mc_value_table, sarsa_value_table_0))
			mse_memo_1.append(mse_tables(mc_value_table, sarsa_value_table_1))

		sarsa_value_table_0, sarsa_reps_table_0 = learn_sarsa_episode_table(
			env1, state_space, action_space, sarsa_value_table_0, sarsa_reps_table_0, n_zero, lambda_value=0)

		sarsa_value_table_1, sarsa_reps_table_1 = learn_sarsa_episode_table(
			env1, state_space, action_space, sarsa_value_table_1, sarsa_reps_table_1, n_zero, lambda_value=1)

	# MSE over lambda

	lambdas = []
	mse_memo_lambda = []

	for lambda_val in tqdm(np.linspace(0, 1, 10), desc='Sarsa(lambda) Learning, part 2/2'):
		lambda_val = np.round(lambda_val, 1)  # for some reason these numbers aren't always exactly round
		sarsa_value_table = LookupTableGeneric(state_space, action_space)
		sarsa_reps_table = LookupTableGeneric(state_space, action_space)
		for _ in range(1000):  # hardcoded since that's the required numebr of learning episodes per lambda value
			sarsa_value_table, sarsa_reps_table = learn_sarsa_episode_table(
				env1, state_space, action_space, sarsa_value_table, sarsa_reps_table, n_zero, lambda_val)
		lambdas.append(lambda_val)
		mse_memo_lambda.append(mse_tables(mc_value_table, sarsa_value_table))

	# plotting
	plotting.plot_sarsa_mse_vs_nsteps(mse_memo_0, mse_memo_1, time_steps, save_plots)
	plotting.plot_sarsa_mse_vs_lambda(mse_memo_lambda, lambdas, save_plots)


def run_sarsa_lfa(num_episodes, alpha, epsilon, mc_value_table, save_plots=True):
	assert num_episodes > 1000, 'Number of episodes must be greater than 1000'
	env1 = Easy21()
	state_space = env1.get_state_space()
	action_space = env1.get_action_space()
	weights_shape = (36, 1)



	# MSE over time for lambda = 0 and lambda = 1

	weights_0 = np.zeros(weights_shape)
	weights_1 = np.zeros(weights_shape)

	time_steps = []
	mse_memo_0 = []
	mse_memo_1 = []

	for i in tqdm(range(num_episodes), desc='Sarsa(lambda) Learning using LFA, part 1/2'):
		if i % 1000 == 0 and not i == 0:
			time_steps.append(i)
			mse_memo_0.append(mse_lfa_table(mc_value_table, weights_0, phi_fn, state_space, action_space))
			mse_memo_1.append(mse_lfa_table(mc_value_table, weights_1, phi_fn, state_space, action_space))

		weights_0 = learn_sarsa_episode_lfa(
			env1, state_space, action_space, phi_fn, weights_0, alpha, epsilon, lambda_value=0)

		weights_1 = learn_sarsa_episode_lfa(
			env1, state_space, action_space, phi_fn, weights_1, alpha, epsilon, lambda_value=1)

	# MSE over lambda

	lambdas = []
	mse_memo_lambda = []

	for lambda_val in tqdm(np.linspace(0, 1, 10), desc='Sarsa(lambda) Learning using LFA, part 2/2'):
		lambda_val = np.round(lambda_val, 1)  # for some reason these numbers aren't always exactly round
		weights = np.zeros(weights_shape)
		for _ in range(1000):  # hardcoded since that's the required numebr of learning episodes per lambda value
			weights = learn_sarsa_episode_lfa(
				env1, state_space, action_space, phi_fn, weights, alpha, epsilon, lambda_value=lambda_val)
		lambdas.append(lambda_val)
		mse_memo_lambda.append(mse_lfa_table(mc_value_table, weights, phi_fn, state_space, action_space))

	# plotting
	plotting.plot_sarsa_mse_vs_nsteps(mse_memo_0, mse_memo_1, time_steps, save_plots, lfa=True)
	plotting.plot_sarsa_mse_vs_lambda(mse_memo_lambda, lambdas, save_plots, lfa=True)


def main():
	# parameters
	number_of_monte_carlo_episodes = 10 ** 7
	number_of_sarsa_episodes = 10 ** 5
	n0 = 100
	alpha_value = 0.01
	epsilon_value = 0.05

	save_plots_to_file = True

	# task 2 - Monte-Carlo control in Easy21
	# run_monte_carlo(number_of_monte_carlo_episodes, n0, save_plots=save_plots_to_file)

	# task 3 - TD Learning in Easy21
	mc_val_tab = pickle.load(open('dumps/mc_table.pkl', 'rb'), encoding=sys.stdout.encoding)
	# run_sarsa_table(number_of_sarsa_episodes, n0, mc_val_tab, save_plots_to_file)

	# task 4 - Linear Function Approximation in Easy21
	run_sarsa_lfa(number_of_sarsa_episodes, alpha_value, epsilon_value, mc_val_tab, save_plots_to_file)


if __name__ == '__main__':
	main()
