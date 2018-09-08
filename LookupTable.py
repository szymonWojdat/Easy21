import numpy as np


class LookupTable:
	_tab = None

	def __init__(self):
		self._tab = np.array([[[[0, 0], [0, 0]]] * 21] * 21)

	@staticmethod
	def _get_action_index(action):
		if action == 'hit':
			action_index = 0
		elif action == 'stick':
			action_index = 1
		else:
			raise ValueError('Error: action should be either \'hit\' or \'stick\'')
		return action_index

	def get_q(self, player, dealer, action):
		action_index = self._get_action_index(action)
		return self._tab[player-1][dealer-1][action_index][0]

	def get_n(self, player, dealer, action):
		action_index = self._get_action_index(action)
		return self._tab[player-1][dealer-1][action_index][1]

	def set_q(self, player, dealer, action, value):
		action_index = self._get_action_index(action)
		self._tab[player - 1][dealer - 1][action_index][0] = value

	def set_n(self, player, dealer, action, value):
		action_index = self._get_action_index(action)
		self._tab[player - 1][dealer - 1][action_index][1] = value

	def inc_n(self, player, dealer, action):
		action_index = self._get_action_index(action)
		self._tab[player - 1][dealer - 1][action_index][1] += 1

	def pprint(self):
		for i, player in enumerate(self._tab):
			print('player = {}:'.format(i+1))
			for j, dealer in enumerate(player):
				print('dealer = {}: hit: {}, stick: {}'.format(j+1, dealer[0], dealer[1]))
			print('\n')
