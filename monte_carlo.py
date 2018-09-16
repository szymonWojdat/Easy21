from common import eps_greedy


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
