from functions.common import eps_greedy
from classes.LookupTable import LookupTableGeneric
import numpy as np


def learn_sarsa_episode_table(env, states, actions, s_a_values, s_a_reps, n0, lambda_value):
	"""
	Executes one episode of Sarsa(lambda) learning using lookup tables
	:param env: environment to use
	:param states: state space of env
	:param actions: action space of env
	:param s_a_values: state-action value function (lookup table)
	:param s_a_reps: state-action counts
	:param n0: influences epsilon in epsilon-greedy policy
	:param lambda_value: lambda parameter value for Sarsa(lambda)
	:return: updated state-action value function, state-action counts and eligibility traces
	"""
	state_memory = []
	observation = env.reset()
	s_a_et = LookupTableGeneric(states, actions)
	done = False

	# pick an episilon-greedy action
	greedy_action = s_a_values.get_greedy_action(observation)
	ns = 0
	for action in actions:
		ns += s_a_reps.get(observation, action)
	epsilon = n0 / (n0 + ns)
	action = eps_greedy(actions, greedy_action, epsilon)
	s_a_reps.increment(observation, action)  # update N
	state_memory.append((observation, action))

	while not done:
		observation_prime, reward, done = env.step(action)

		# pick an episilon-greedy action
		greedy_action = s_a_values.get_greedy_action(observation_prime)
		ns = 0
		for action in actions:
			ns += s_a_reps.get(observation, action)
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


def learn_sarsa_episode_lfa(env, states, actions, phi: function, theta: np.array, alpha, epsilon, lambda_value):
	"""
	Executes one episode of Sarsa(lambda) learning using linear function approximation
	:param env: environment to use
	:param states: state space of env
	:param actions: action space of env
	:param phi: function that maps (s, a) pair to a binary feature vector
	:param theta: vector of weights used in linear fn approx.
	:param alpha: learning rate/step size
	:param epsilon: used in epsilon-greedy policy
	:param lambda_value: lambda parameter value for Sarsa(lambda)
	:return: updated state-action value function, state-action counts and eligibility traces
	"""
	state_memory = []
	observation = env.reset()
	s_a_et = LookupTableGeneric(states, actions)
	done = False

	def q(s, a, th):
		return np.matmul(phi(s, a), th)

	def get_eps_greedy_action(s, th):
		action_values = {}
		for act in actions:
			action_values[act] = q(s, act, th)
		best_actions = [k for k, v in action_values.items() if v == max(action_values.values())]
		greedy_action = np.random.choice(best_actions)
		return eps_greedy(actions, greedy_action, epsilon)

	action = get_eps_greedy_action(observation, theta)
	state_memory.append((observation, action))

	while not done:
		observation_prime, reward, done = env.step(action)
		action_prime = get_eps_greedy_action(observation_prime, theta)

		# calculate the TD-error (no discounting!) and increment eligibility trace for current S-A pair
		delta = alpha * (reward + q(observation_prime, action_prime, theta) - q(observation, action, theta))
		s_a_et.increment(observation, action)

		for _observation, _action in state_memory:
			# update q (effectively theta) in the direction of delta (TD error)
			q_updated = q(_observation, _action, theta) + delta
			theta = np.matmul(np.linalg.inv(phi(_observation, action)), q_updated)
			et_update = lambda_value * s_a_et.get(_observation, _action)
			s_a_et.set(_observation, _action, et_update)

		observation, action = observation_prime, action_prime
		state_memory.append((observation, action))

	return theta
