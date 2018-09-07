from random import choice

CARDS = list(range(1, 11)) + list(range(1, 11)) + list(range(-10, 0))


class Easy21:
	_done = None
	_dealer = None
	_player = None

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
		else:
			raise ValueError('Error: action should be either \'hit\' or \'stick\'')
		return {'player': self._player, 'dealer': self._dealer}, reward, self._done

	def reset(self):
		self._dealer = self._draw(black_only=True)
		self._player = self._draw(black_only=True)
		self._done = False
		return {'player': self._player, 'dealer': self._dealer}
