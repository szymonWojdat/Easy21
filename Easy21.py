from random import choice

CARDS = list(range(1, 11)) + list(range(1, 11)) + list(range(-10, 0))


class Easy21:
	_done = None
	_dealer = None
	_player = None

	# player can technically reach a sum up to 31, not that the game is played then but we need it for the lookahead
	# _state_space = tuple([(player, dealer) for player in range(-9, 32) for dealer in range(1, 11)])
	_state_space = tuple([(player, dealer) for player in range(-9, 32) for dealer in range(-9, 32)])
	_action_space = ('hit', 'stick')

	@staticmethod
	def _draw(black_only=False):
		global CARDS
		number = choice(CARDS)
		if black_only:
			return abs(number)
		else:
			return number

	@staticmethod
	def _is_busted(n):
		return not 1 <= n <= 21

	@staticmethod
	def _dealer_draws(n):
		return 1 <= n <= 16

	def step(self, action):
		reward = 0
		assert action in self._action_space,\
			'Error: action {} should be in action space: {}'.format(action, self._action_space)
		if action == 'stick':  # dealer makes their moves
			while self._dealer_draws(self._dealer):
				self._dealer += self._draw()
			if self._player > self._dealer or self._is_busted(self._dealer):
				reward = 1
			elif self._player < self._dealer:
				reward = -1
			self._done = True
		elif action == 'hit':  # player draws a card
			self._player += self._draw()
			if self._is_busted(self._player):
				reward = -1
				self._done = True
		state = (self._player, self._dealer)
		assert state in self._state_space,\
			'Error: state {} should be in state space: {}'.format(state, self._state_space)
		return state, reward, self._done

	def reset(self):
		self._dealer = self._draw(black_only=True)
		self._player = self._draw(black_only=True)
		self._done = False
		return self._player, self._dealer

	def get_state_space(self):
		return self._state_space

	def get_action_space(self):
		return self._action_space
