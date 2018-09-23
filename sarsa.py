from common import eps_greedy
from LookupTable import LookupTableGeneric


def learn_sarsa_episode(env, states, actions, s_a_values, s_a_reps, n0, lambda_value):
	"""
	Executes one episode of Sarsa(lambda) learning
	:param env: environment to use
	:param actions: action space of env
	:param s_a_values: state-action value function (lookup table)
	:param s_a_reps: state-action counts
	:param n0: influences epsilon in epsilon-greedy policy
	:param lambda_value: lambda parameter value for Sarsa(lambda)
	:return: updated state-action value function, state-action counts and eligibility traces
	"""
	total_reward = 0
	state_memory = []
	observation = env.reset()
	# player, dealer = observation
	s_a_et = LookupTableGeneric(states, actions)
	done = False

	# pick an episilon-greedy action
	greedy_action = s_a_values.get_greedy_action(observation)
	ns = s_a_reps.get(observation, 'hit') + s_a_reps.get(observation, 'stick')  # TODO - make this generic
	epsilon = n0 / (n0 + ns)
	action = eps_greedy(actions, greedy_action, epsilon)
	s_a_reps.increment(observation, action)  # update N
	state_memory.append((observation, action))

	while not done:
		observation_prime, reward, done = env.step(action)
		# player_prime, dealer_prime = observation
		total_reward += reward

		# pick an episilon-greedy action
		greedy_action = s_a_values.get_greedy_action(observation_prime)
		ns = s_a_reps.get(observation_prime, 'hit') + s_a_reps.get(observation, 'stick')  # TODO - make this generic
		epsilon = n0 / (n0 + ns)
		action_prime = eps_greedy(actions, greedy_action, epsilon)

		# calculate the TD-error (no discounting!) and increment eligibility trace for current S-A pair
		delta = reward + s_a_values.get(observation_prime, action_prime) -\
			s_a_values.get(observation, action)
		s_a_et.increment(observation, action)

		for _observation, _action in state_memory:
			alpha = 1 / s_a_reps.get(_observation, _action)
			q_update = alpha * delta * s_a_et.get(_observation, _action)
			s_a_values.update(_observation, _action, q_update)  # update Q
			et_update = lambda_value * s_a_et.get(_observation, _action)
			s_a_et.set(_observation, _action, et_update)

		observation, action = observation_prime, action_prime
		state_memory.append((observation, action))
		s_a_reps.increment(observation, action)

	return s_a_values, s_a_reps
