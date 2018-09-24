import numpy as np


class LookupTable:
	_tab = None
	_action_space = None

	def __init__(self, action_space):
		self._tab = np.array([[[.0, .0]] * 31] * 31)
		self._action_space = action_space

	def _get_action_index(self, action):
		assert action in self._action_space, \
			'Action {} is not a part of action space: {}'.format(action, self._action_space)
		return self._action_space.index(action)

	def get(self, player, dealer, action):
		action_index = self._get_action_index(action)
		return self._tab[player-1][dealer-1][action_index]

	def update(self, player, dealer, action, value):
		action_index = self._get_action_index(action)
		self._tab[player - 1][dealer - 1][action_index] += value

	def set(self, player, dealer, action, value):
		action_index = self._get_action_index(action)
		self._tab[player - 1][dealer - 1][action_index] = value

	def increment(self, player, dealer, action):
		action_index = self._get_action_index(action)
		self._tab[player - 1][dealer - 1][action_index] += 1

	def get_greedy_action(self, player, dealer):
		# probably possible to do in one line but idc for now
		action_values = {}
		for action in self._action_space:
			action_values[action] = self.get(player, dealer, action)
		a = [k for k, v in action_values.items() if v == max(action_values.values())]
		return np.random.choice(a)


class LookupTableGeneric:
	_tab = None
	_state_space = None
	_action_space = None

	def __init__(self, state_space, action_space):
		self._state_space = state_space
		self._action_space = action_space
		self._tab = np.array([[.0] * len(action_space)] * len(state_space))

	def _get_state_index(self, state):
		assert state in self._state_space,\
			'State {} is not a part of state space: {}'.format(state, self._state_space)
		return self._state_space.index(state)

	def _get_action_index(self, action):
		assert action in self._action_space,\
			'Action {} is not a part of action space: {}'.format(action, self._action_space)
		return self._action_space.index(action)

	def get(self, state, action):
		state_index = self._get_state_index(state)
		action_index = self._get_action_index(action)
		return self._tab[state_index][action_index]

	def update(self, state, action, value):
		state_index = self._get_state_index(state)
		action_index = self._get_action_index(action)
		self._tab[state_index][action_index] += value

	def set(self, state, action, value):
		state_index = self._get_state_index(state)
		action_index = self._get_action_index(action)
		self._tab[state_index][action_index] = value

	def increment(self, state, action):
		state_index = self._get_state_index(state)
		action_index = self._get_action_index(action)
		self._tab[state_index][action_index] += 1

	def get_greedy_action(self, state):
		# probably possible to do in one line but idc for now
		action_values = {}
		for action in self._action_space:
			action_values[action] = self.get(state, action)
		a = [k for k, v in action_values.items() if v == max(action_values.values())]
		return np.random.choice(a)
