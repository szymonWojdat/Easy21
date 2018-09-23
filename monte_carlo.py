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
	done = False
	observation = env.reset()

	while not done:
		# pick an episilon-greedy action
		greedy_action = s_a_values.get_greedy_action(observation)
		ns = 0
		for action in actions:
			ns += s_a_reps.get(observation, action)
		epsilon = n0/(n0 + ns)
		action = eps_greedy(actions, greedy_action, epsilon)
		s_a_reps.increment(observation, action)  # update N
		state_memory.append((observation, action))

		observation, reward, done = env.step(action)
		total_reward += reward

	for observation, action in state_memory:
		alpha = 1/s_a_reps.get(observation, action)
		delta_q = alpha * (total_reward - s_a_values.get(observation, action))
		s_a_values.update(observation, action, delta_q)  # update Q

	return s_a_values, s_a_reps
