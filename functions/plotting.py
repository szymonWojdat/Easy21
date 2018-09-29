import matplotlib.pyplot as plt


def plot_value_for_player_dealer(value_table, action_space):
	player = []
	dealer = []
	value = []
	# states, for which agent makes a decision and then they get updated:
	actual_state_space = [(player, dealer) for player in range(10, 21) for dealer in reversed(range(1, 11))]
	for state in actual_state_space:
		action_values = []
		for action in action_space:
			action_values.append(value_table.get(state, action))
		player.append(state[0])
		dealer.append(state[1])
		value.append(max(action_values))

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_trisurf(player, dealer, value, linewidth=0.2, cmap=plt.cm.viridis)
	fig.tight_layout()
	ax.set(
		xlabel='player sum',
		ylabel='dealer card',
		zlabel='value (Q)'
	)
	ax.view_init(30, 210)  # I was doing my best to make the angle similar to the one in Sutton/Barto's book :)

	plt.show()
	fig.savefig('graphs/mc_value_function.png')


def plot_sarsa_mse_vs_nsteps(mse_memo_0, mse_memo_1, time_steps):
	fig1, ax1 = plt.subplots()
	ax1.plot(time_steps, mse_memo_0, label='lambda = 0')
	ax1.plot(time_steps, mse_memo_1, label='lambda = 1')
	ax1.set(
		xlabel='Number of steps',
		ylabel='MSE',
		title='Sarsa(lambda) Mean-Squared Error Over Time'
	)
	ax1.legend(loc='best')

	plt.show()
	fig1.savefig('graphs/sarsa_mse_over_time.png')


def plot_sarsa_mse_vs_lambda(mse_memo_lambda, lambdas):

	fig2, ax2 = plt.subplots()
	ax2.plot(lambdas, mse_memo_lambda)
	ax2.set(
		xlabel='Lambda',
		ylabel='MSE',
		title='Sarsa(lambda) MSE over Lambda, 1000 Episodes per Value'
	)

	plt.show()
	fig2.savefig('graphs/sarsa_mse_over_lambda.png')
