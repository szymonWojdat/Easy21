import numpy as np


class LookupTable:
	_tab = None
	_action_space = None

	def __init__(self, action_space):
		self._tab = np.array([[[.0, .0]] * 21] * 21)
		self._action_space = action_space

	def _get_action_index(self, action):
		if action not in self._action_space:
			raise ValueError('Error: action should be either of those: {}'.format(self._action_space))
		else:
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

	# def pprint(self):
	# 	for i, player in enumerate(self._tab):
	# 		print('player = {}:'.format(i+1))
	# 		for j, dealer in enumerate(player):
	# 			print('dealer = {}: hit: {}, stick: {}'.format(j+1, dealer[0], dealer[1]))
	# 		print('\n')

	def get_greedy_action(self, player, dealer):
		# probably possible to do in one line but idc for now
		action_values = {}
		for action in self._action_space:
			action_values[action] = self.get(player, dealer, action)
		return np.random.choice([k for k, v in action_values.items() if v == max(action_values.values())])
