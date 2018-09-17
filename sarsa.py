from common import eps_greedy
from LookupTable import LookupTable


def learn_sarsa_episode(env, actions, s_a_values, s_a_reps, n0, lambda_value):
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
	player, dealer = observation
	s_a_et = LookupTable(actions)
	done = False

	# pick an episilon-greedy action
	greedy_action = s_a_values.get_greedy_action(player, dealer)
	ns = s_a_reps.get(player, dealer, 'hit') + s_a_reps.get(player, dealer, 'stick')
	epsilon = n0 / (n0 + ns)
	action = eps_greedy(actions, greedy_action, epsilon)
	s_a_reps.increment(player, dealer, action)  # update N
	state_memory.append(((player, dealer), action))

	while not done:
		observation, reward, done = env.step(action)
		player_prime, dealer_prime = observation
		total_reward += reward

		# pick an episilon-greedy action
		greedy_action = s_a_values.get_greedy_action(player_prime, dealer_prime)
		ns = s_a_reps.get(player_prime, dealer_prime, 'hit') + s_a_reps.get(player, dealer, 'stick')
		epsilon = n0 / (n0 + ns)
		action_prime = eps_greedy(actions, greedy_action, epsilon)

		# calculate the TD-error (no discounting!) and increment eligibility trace for current S-A pair
		delta = reward + s_a_values.get(player_prime, dealer_prime, action_prime) -\
			s_a_values.get(player, dealer, action)
		s_a_et.increment(player, dealer, action)

		for _observation, _action in state_memory:
			_player, _dealer = _observation
			alpha = 1 / s_a_reps.get(_player, _dealer, _action)
			q_update = alpha * delta * s_a_et.get(_player, _dealer, _action)
			s_a_values.update(_player, _dealer, _action, q_update)  # update Q
			et_update = lambda_value * s_a_et.get(_player, _dealer, _action)
			s_a_et.set(_player, _dealer, _action, et_update)

		player, dealer, action = player_prime, dealer_prime, action_prime
		state_memory.append(((player, dealer), action))
		s_a_reps.increment(player, dealer, action)

	return s_a_values, s_a_reps
