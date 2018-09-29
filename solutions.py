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
from mpl_toolkits.mplot3d import Axes3D


def run_monte_carlo(num_episodes, n_zero):
	env1 = Easy21()
	state_space = env1.get_state_space()
	action_space = env1.get_action_space()

	mc_value_table = LookupTableGeneric(state_space, action_space)
	mc_reps_table = LookupTableGeneric(state_space, action_space)

	# learning
	for i in range(num_episodes):
		mc_value_table, mc_reps_table = learn_mc_episode(env1, action_space, mc_value_table, mc_reps_table, n_zero)

	# plotting

	player = []
	dealer = []
	value = []
	for state in state_space:
		action_values = []
		for action in action_space:
			action_values.append(mc_value_table.get(state, action))
		player.append(state[0])
		dealer.append(state[1])
		value.append(max(action_values))

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_trisurf(player, dealer, value, linewidth=0.2, antialiased=True)
	fig.tight_layout()
	ax.set(
		xlabel='player card',
		ylabel='dealer card',
		zlabel='value'
		# title='Monte-Carlo State-Value Function'
	)
	ax.view_init(22.5, 225)

	plt.show()

	# saving to pickle file

	# directory = 'dumps'
	# if not os.path.exists(directory):
	# 	os.makedirs(directory)
	# with open(directory + '/mc_table.pkl', 'wb') as file:
	# 	pickle.dump(mc_value_table, file)


def run_sarsa(num_episodes, n_zero, mc_value_table):
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

	for i in range(num_episodes):
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

	for lambda_val in np.linspace(0, 1, 10):
		lambda_val = np.round(lambda_val, 1)  # for some reason these numbers aren't always exactly round
		sarsa_value_table = LookupTableGeneric(state_space, action_space)
		sarsa_reps_table = LookupTableGeneric(state_space, action_space)
		for _ in range(1000):  # hardcoded since that's the required numebr of learning episodes per lambda value
			sarsa_value_table, sarsa_reps_table = learn_sarsa_episode(
				env1, state_space, action_space, sarsa_value_table, sarsa_reps_table, n_zero, lambda_val)
		lambdas.append(lambda_val)
		mse_memo_lambda.append(mse(mc_value_table, sarsa_value_table))

	# plotting

	fig1, ax1 = plt.subplots()
	ax1.plot(time_steps, mse_memo_0, label='lambda = 0')
	ax1.plot(time_steps, mse_memo_1, label='lambda = 1')
	ax1.set(
		xlabel='Number of steps',
		ylabel='MSE',
		title='Sarsa(lambda) Mean-Squared Error Over Time'
	)
	ax1.legend(loc='best')

	fig2, ax2 = plt.subplots()
	ax2.plot(lambdas, mse_memo_lambda)
	ax2.set(
		xlabel='Lambda',
		ylabel='MSE',
		title='Sarsa(lambda) MSE over Lambda, 1000 Episodes per Value'
	)

	plt.show()
	fig1.savefig('graphs/sarsa_mse_over_time.png')
	fig2.savefig('graphs/sarsa_mse_over_lambda.png')


def main():
	# parameters
	n_learn_ep = 10 ** 4
	n0 = 100

	# task 2 - Monte-Carlo control in Easy21
	run_monte_carlo(n_learn_ep, n0)

	# task 3 - TD Learning in Easy21
	# mc_val_tab = pickle.load(open('dumps/mc_table.pkl', 'rb'), encoding=sys.stdout.encoding)
	# run_sarsa(n_learn_ep, n0, mc_val_tab)


if __name__ == '__main__':
	main()
