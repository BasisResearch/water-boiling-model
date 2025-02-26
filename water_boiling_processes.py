from typing import Dict, List
from collections import namedtuple
import random

AbstractState = namedtuple('State', [
    'boiling', 'pot_location', 'stove_on', 'faucet_on', 'pot_filled',
    'water_spilled', 'action'
])


# Abstract class for cause-effect relations
# When the condition holds, the effect is scheduled for a random time in the
# future
class CausalProcess:

    def __init__(self, name, delay_distribution):
        self.name = name
        self.delay_distribution = delay_distribution

    def condition(self, history):
        raise NotImplementedError

    def effect(self, state):
        raise NotImplementedError


# Two different distributions for cause-effect delays
class ConstantDelay:

    def __init__(self, delay):
        self.delay = delay

    def sample(self):
        return self.delay


class GaussianDelay:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        while True:
            delay = int(random.gauss(self.mean, self.std) + 0.5)
            if delay > 0:
                return delay


# Example processes for the boiling water environment


# First, causal processes that describe our own action space
class ToggleFaucet(CausalProcess):

    def __init__(self):
        super().__init__('ToggleFaucet', GaussianDelay(5, 2))

    def condition(self, history):

        def check(s):
            return s.action == "toggle_faucet"

        return check(history[-1]) and (len(history) == 1
                                       or not check(history[-2]))

    def effect(self, state):
        return state._replace(faucet_on=not state.faucet_on, action=None)


class ToggleStove(CausalProcess):

    def __init__(self):
        super().__init__('ToggleStove', GaussianDelay(5, 2))

    def condition(self, history):

        def check(s):
            return s.action == "toggle_stove"

        return check(history[-1]) and (len(history) == 1
                                       or not check(history[-2]))

    def effect(self, state):
        return state._replace(stove_on=not state.stove_on, action=None)


class MoveToFaucet(CausalProcess):

    def __init__(self):
        super().__init__('MoveToFaucet', GaussianDelay(5, 2))

    def condition(self, history):

        def check(s):
            return s.action == "move_to_faucet"

        return check(history[-1]) and (len(history) == 1
                                       or not check(history[-2]))

    def effect(self, state):
        return state._replace(pot_location="faucet", action=None)


class MoveToStove(CausalProcess):

    def __init__(self):
        super().__init__('MoveToStove', GaussianDelay(5, 2))

    def condition(self, history):

        def check(s):
            return s.action == "move_to_stove"

        return check(history[-1]) and (len(history) == 1
                                       or not check(history[-2]))

    def effect(self, state):
        return state._replace(pot_location="stove", action=None)


# New Noop action process
class Noop(CausalProcess):

    def __init__(self):
        super().__init__('Noop',
                         GaussianDelay(20, 1))  # Random delay from Gaussian

    def condition(self, history):

        def check(s):
            return s.action == "noop"

        return check(history[-1])

    def effect(self, state):
        return state._replace(action=None)


# then, causal processes that we don't get to control, and which are part of the
# environments passive dynamics
class FillPot(CausalProcess):

    def __init__(self):
        super().__init__('FillPot', GaussianDelay(5, 2))

    def condition(self, history):

        def check(s):
            return s.pot_location == "faucet" and s.faucet_on and \
                not s.pot_filled

        return check(history[-1]) and (len(history) == 1
                                       or not check(history[-2]))

    def effect(self, state):
        return state._replace(pot_filled=True)


class OverfillPot(CausalProcess):

    def __init__(self):
        super().__init__('OverfillPot', ConstantDelay(1))

    def condition(self, history):

        def check(s):
            return s.pot_location == "faucet" and s.faucet_on and \
                s.pot_filled and not s.water_spilled

        suffix_passing_check = [check(s) for s in history[::-1]].index(False)

        if suffix_passing_check > 10 + random.randint(
                0, 10):  # this can be made probabilistic
            return True
        else:
            return False

    def effect(self, state):
        return state._replace(water_spilled=True)


class Boil(CausalProcess):

    def __init__(self):
        super().__init__('Boil', ConstantDelay(1))

    def condition(self, history):

        def check(s):
            return s.pot_location == "stove" and s.stove_on and \
                s.pot_filled and not s.boiling

        suffix_passing_check = [check(s) for s in history[::-1]].index(False)

        return suffix_passing_check > 10 + random.randint(
            0, 30)  # this can be made probabilistic

    def effect(self, state):
        return state._replace(boiling=True)
