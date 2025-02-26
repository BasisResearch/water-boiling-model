"""An example of a world model that an agent may potentially learn.

It allows the agent to generate a sequence of imaginary states given at sequence
of imaginary action.
The effects of actions such as ToggleFaucet or MoveToFaucet happens after a
delay after their conditions are met. This is also true for processes such as
FillPot.
For processes such as OverfillPot or Boil, the effects happen immediately after
the condition holds for a certain number of time steps.0
"""
from collections import namedtuple
import random

random.seed(0)

State = namedtuple('State', [
    'boiling', 'pot_location', 'stove_on', 'faucet_on', 'pot_filled',
    'water_spilled', 'action'
])


# Abstract class for cause-effect relations
# When the condition holds, the effect is scheduled for a random time in the future
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


# The environment keeps track of the state, history, and cause-effects whose condition has been triggered but whose effect has not yet "landed", meaning they are scheduled for the future
class Env:

    def __init__(self, processes, initial_state):
        self.processes = processes
        self.state = initial_state
        self.history = [initial_state]
        self.scheduled_events = {}
        self.t = 0

    def small_step(self, action=None):
        initial_state = self.state

        # 1. Process the action
        if action is not None:
            # action is of the form do(var)=val
            variable, value = action
            self.state = self.state._replace(**{variable: value})
            print(f"At time {self.t}, do({variable}={value})")

        # 2. Process any events scheduled for this timestep
        if self.t in self.scheduled_events:
            for process in self.scheduled_events[self.t]:
                self.state = process.effect(self.state)
            del self.scheduled_events[self.t]

        self.history.append(self.state)

        # 3. Schedule new events
        for process in self.processes:
            if process.condition(self.history):
                delay = process.delay_distribution.sample()
                schedule_time = self.t + delay
                print(
                    f"At time {self.t}, {process.name}.effect scheduled for "
                    f"{schedule_time}"
                )
                if schedule_time not in self.scheduled_events:
                    self.scheduled_events[schedule_time] = list()
                self.scheduled_events[schedule_time].append(process)

        # Check if state changed
        if self.state != initial_state:
            changed = []
            for var in self.state._fields:
                if getattr(self.state, var) != getattr(initial_state, var):
                    changed.append(f"{var}={getattr(self.state, var)}")
            print(f"At time {self.t}, state changes to {', '.join(changed)}")

            # remove all the scheduled wait event
            state_change = [
                getattr(self.state, var) != getattr(initial_state, var)
                for var in self.state._fields if var != 'action'
            ]
            if any(state_change):
                for t in list(self.scheduled_events.keys()):
                    if self.scheduled_events[t][0].name == 'Noop':
                        del self.scheduled_events[t]
                    # change action to None
                    self.state = self.state._replace(action=None)

        self.t += 1

    def big_step(self, action=None):
        initial_state = self.state
        while self.state == initial_state:
            self.small_step(action)
            if action is not None:
                # update the initial state and set action to None
                initial_state = self.state
                action = None
        print(
            f"Stopping big step because state changed from this:\n"
            f"\t{initial_state}\nto this:\n\t{self.state}"
        )
        return self.state


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


env = Env(
    [
        ToggleFaucet(),
        ToggleStove(),
        MoveToFaucet(),
        MoveToStove(),
        Noop(),
        FillPot(),
        OverfillPot(),
        Boil()
    ],
    # initial state
    State(boiling=False,
          pot_location="table",
          stove_on=False,
          faucet_on=False,
          pot_filled=False,
          water_spilled=False,
          action=None))




if __name__ == "__main__":

    # a policy that can be run at any time point and as the right thing
    def small_step_policy(history):
        if len(history) == 1:
            return "action", "move_to_faucet"

        state = history[-1]

        if state.pot_location == "faucet" and not state.faucet_on and \
            not state.pot_filled and state.action != "toggle_faucet":
            return "action", "toggle_faucet"

        if state.pot_location == "faucet" and state.faucet_on and state.pot_filled \
            and state.action != "toggle_faucet":
            return "action", "toggle_faucet"

        if state.pot_location == "faucet" and state.pot_filled and \
            not state.faucet_on and state.action != "move_to_stove":
            return "action", "move_to_stove"

        if state.pot_location == "stove" and not state.stove_on and \
            state.action != "toggle_stove":
            return "action", "toggle_stove"

    # a policy that is more of a plan and which should be run in between big 
    # steps
    big_step_plan = [
        ("action", "move_to_faucet"),
        ("action", "toggle_faucet"),
        ("action", "noop"),
        # None,  # wait for the faucet to fill the pot
        # note that if you put this extra None here, we will keep on waiting 
        # after the pot is filled, which causes it to spill
        # boiling still works but the final state has water_spilled=True
        # None,
        ("action", "toggle_faucet"),
        ("action", "move_to_stove"),
        ("action", "toggle_stove"),
        ("action", "noop"),
        # None  # wait for it to boil
    ]

    def big_step_policy(history):
        global big_step_plan
        del history
        if len(big_step_plan) > 0:
            return big_step_plan.pop(0)
        else:
            return None

    for _ in range(100):
        action = big_step_policy(env.history)
        print("\n" * 3)
        if action is not None:
            print(f"At time {env.t}, {action[0]}={action[1]}")
        else:
            print(f"At time {env.t}, action=NONE")

        state = env.big_step(action)
        if state.boiling:
            print("\nWater is boiling!")
            break
    # show the history now that we are done
    print("\nHistory:")
    for state in env.history:
        print(state)
