from classes.Easy21 import Easy21
from classes.LookupTable import LookupTableGeneric
import numpy as np
from functions.monte_carlo import learn_mc_episode
from functions.sarsa import learn_sarsa_episode
from functions.common import mse
import matplotlib.pyplot as plt
import pickle
import os
import sys


def run_monte_carlo(num_episodes, n_zero):
	env1 = Easy21()
	state_space = env1.get_state_space()
	action_space = env1.get_action_space()

	mc_value_table = LookupTableGeneric(state_space, action_space)
	mc_reps_table = LookupTableGeneric(state_space, action_space)

	# learning
	for i in range(num_episodes):
		mc_value_table, mc_reps_table = learn_mc_episode(env1, action_space, mc_value_table, mc_reps_table, n_zero)

	# TODO - add plotting here

	directory = 'dumps'
	if not os.path.exists(directory):
		os.makedirs(directory)
	with open(directory + '/mc_table.pkl', 'wb') as file:
		pickle.dump(mc_value_table, file)


def run_sarsa(num_episodes, n_zero, mc_value_table):
	env1 = Easy21()
	state_space = env1.get_state_space()
	action_space = env1.get_action_space()

	# MSE over time for lambda = 0 and lambda = 1

	sarsa_value_table_0 = LookupTableGeneric(state_space, action_space)
	sarsa_reps_table_0 = LookupTableGeneric(state_space, action_space)
	sarsa_value_table_1 = LookupTableGeneric(state_space, action_space)
	sarsa_reps_table_1 = LookupTableGeneric(state_space, action_space)

	mse_memo_0 = []
	mse_memo_1 = []

	for i in range(num_episodes):
		if i % 1000 == 0 and not i == 0:
			mse_memo_0.append((i, mse(mc_value_table, sarsa_value_table_0)))
			mse_memo_1.append((i, mse(mc_value_table, sarsa_value_table_1)))

		sarsa_value_table_0, sarsa_reps_table_0 = learn_sarsa_episode(
			env1, state_space, action_space, sarsa_value_table_0, sarsa_reps_table_0, n_zero, lambda_value=0)

		sarsa_value_table_1, sarsa_reps_table_1 = learn_sarsa_episode(
			env1, state_space, action_space, sarsa_value_table_1, sarsa_reps_table_1, n_zero, lambda_value=1)

	# iterating over different lambda values - 0, 0.1, ..., 1

	mse_memo_lambda = []

	for lambda_val in np.linspace(0, 1, 10):
		lambda_val = np.round(lambda_val, 1)  # for some reason these numbers aren't always exactly round
		sarsa_value_table = LookupTableGeneric(state_space, action_space)
		sarsa_reps_table = LookupTableGeneric(state_space, action_space)
		for _ in range(1000):  # hardcoded since that's the required numebr of learning episodes per lambda value
			sarsa_value_table, sarsa_reps_table = learn_sarsa_episode(
				env1, state_space, action_space, sarsa_value_table, sarsa_reps_table, n_zero, lambda_val)
		mse_memo_lambda.append((lambda_val, mse(mc_value_table, sarsa_value_table_1)))

	# TODO - add plotting here - MSE vs time - lamdba = 0 and = 1 on the same graph
	# TODO - add plotting here - MSE vs lambda


def main():
	# parameters
	n_learn_ep = 10 ** 5
	n0 = 100

	run_monte_carlo(n_learn_ep, n0)  # task 2 - Monte-Carlo control in Easy21

	mc_val_tab = pickle.load(open('dumps/mc_table.pkl', 'rb'), encoding=sys.stdout.encoding)
	run_sarsa(n_learn_ep, n0, mc_val_tab)  # task 3 - TD Learning in Easy21


if __name__ == '__main__':
	main()
