from collections import namedtuple, deque
import random
from copy import deepcopy

random.seed(0)

State = namedtuple('State', [
    'boiling', 'pot_location', 'stove_on', 'faucet_on', 'pot_filled',
    'water_spilled', 'action'
])


# Abstract class for cause-effect relations
# When the precondition holds, the effect is scheduled for a random time in the future
class CausalProcess:

    def __init__(self, name, delay_distribution):
        self.name = name
        self.delay_distribution = delay_distribution

    def precondition(self, history):
        raise NotImplementedError

    def effect(self, state, history_slice):
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


# The environment keeps track of the state, history, and cause-effects whose precondition has been triggered but whose effect has not yet "landed", meaning they are scheduled for the future
class Env:

    def __init__(self, processes, initial_state):
        self.processes = processes
        self.state = initial_state
        self.history = [initial_state]
        self.scheduled_events = {}
        self.t = 0

    def small_step(self, action=None, verbose=False):
        initial_state = self.state
        if action is not None:
            # action is of the form do(var)=val
            variable, value = action
            self.state = self.state._replace(**{variable: value})
            if verbose: print(f"At time {self.t}, do({variable}={value})")

        # Process any events scheduled for this timestep
        if self.t in self.scheduled_events:
            for process, start_time in self.scheduled_events[self.t]:
                self.state = process.effect(self.state,
                                            self.history[start_time + 1:])
            del self.scheduled_events[self.t]

        self.history.append(self.state)

        # Schedule new events
        for process in self.processes:
            if process.precondition(self.history):
                delay = process.delay_distribution.sample()
                schedule_time = self.t + delay
                if verbose:
                    print(
                        f"At time {self.t}, {process.name}.effect scheduled for {schedule_time}"
                    )
                if schedule_time not in self.scheduled_events:
                    self.scheduled_events[schedule_time] = list()
                self.scheduled_events[schedule_time].append((process, self.t))

        # Check if state changed
        if verbose and self.state != initial_state:
            changed = []
            for var in self.state._fields:
                if getattr(self.state, var) != getattr(initial_state, var):
                    changed.append(f"{var}={getattr(self.state, var)}")
            print(f"At time {self.t}, state changes to {', '.join(changed)}")

        self.t += 1

    def big_step(self, action=None, verbose=False):
        initial_state = self.state
        while self.state == initial_state:
            self.small_step(action, verbose)
            if action is not None:
                initial_state = self.state
                action = None

            if self.state == initial_state and len(self.scheduled_events) == 0:
                if verbose:
                    print(
                        f"Stopping because nothing is going to ever change without further actions"
                    )
                return self.state

        if verbose:
            print(
                f"Stopping big step because state changed from this:\n\t{initial_state}\nto this:\n\t{self.state}"
            )
        return self.state


# Example processes for the boiling water environment


# First, causal processes that describe our own action space
class ToggleFaucet(CausalProcess):

    def __init__(self):
        super().__init__('ToggleFaucet', GaussianDelay(5, 2))

    def precondition(self, history):

        def check(s):
            return s.action == "toggle_faucet"

        return check(history[-1]) and (len(history) == 1
                                       or not check(history[-2]))

    def effect(self, state, history_slice):
        return state._replace(faucet_on=not state.faucet_on, action=None)


class ToggleStove(CausalProcess):

    def __init__(self):
        super().__init__('ToggleStove', GaussianDelay(5, 2))

    def precondition(self, history):

        def check(s):
            return s.action == "toggle_stove"

        return check(history[-1]) and (len(history) == 1
                                       or not check(history[-2]))

    def effect(self, state, history_slice):
        return state._replace(stove_on=not state.stove_on, action=None)


class MoveToFaucet(CausalProcess):

    def __init__(self):
        super().__init__('MoveToFaucet', GaussianDelay(5, 2))

    def precondition(self, history):

        def check(s):
            return s.action == "move_to_faucet"

        return check(history[-1]) and (len(history) == 1
                                       or not check(history[-2]))

    def effect(self, state, history_slice):
        return state._replace(pot_location="faucet", action=None)


class MoveToStove(CausalProcess):

    def __init__(self):
        super().__init__('MoveToStove', GaussianDelay(5, 2))

    def precondition(self, history):

        def check(s):
            return s.action == "move_to_stove"

        return check(history[-1]) and (len(history) == 1
                                       or not check(history[-2]))

    def effect(self, state, history_slice):
        return state._replace(pot_location="stove", action=None)


# then, causal processes that we don't get to control, and which are part of the environments passive dynamics
class FillPot(CausalProcess):

    def __init__(self):
        super().__init__('FillPot', GaussianDelay(25, 2))

    def precondition(self, history):

        def check(s):
            return s.pot_location == "faucet" and s.faucet_on and not s.pot_filled

        return check(history[-1]) and (len(history) == 1
                                       or not check(history[-2]))

    def effect(self, state, history_slice):
        if all(s.pot_location == "faucet" and s.faucet_on for s in
               history_slice):  # make sure that it stayed on the faucet
            return state._replace(pot_filled=True)
        else:
            return state


class OverfillPot(CausalProcess):

    def __init__(self):
        super().__init__('OverfillPot', GaussianDelay(10, 1))

    def precondition(self, history):

        def check(s):
            return s.pot_location == "faucet" and s.faucet_on and s.pot_filled and not s.water_spilled

        return check(history[-1]) and (len(history) == 1
                                       or not check(history[-2]))

    def effect(self, state, history_slice):
        # make sure that the pot was on the faucet and filled
        if all(s.pot_location == "faucet" and s.faucet_on and s.pot_filled
               for s in history_slice):
            return state._replace(water_spilled=True)
        else:
            return state


class Spill(
        CausalProcess
):  # water can spill whenever you're running the faucet and there's no pot to catch the water

    def __init__(self):
        super().__init__('Spill', ConstantDelay(1))

    def precondition(self, history):

        def check(s):
            return s.pot_location != "faucet" and s.faucet_on and not s.water_spilled

        return check(history[-1]) and (len(history) == 1
                                       or not check(history[-2]))

    def effect(self, state, history_slice):
        return state._replace(water_spilled=True)


class Boil(CausalProcess):

    def __init__(self):
        super().__init__('Boil', GaussianDelay(30, 1))

    def precondition(self, history):

        def check(s):
            return s.pot_location == "stove" and s.stove_on and s.pot_filled

        return check(history[-1]) and (len(history) == 1
                                       or not check(history[-2]))

    def effect(self, state, history_slice):
        # make sure that the pot was on the stove and filled
        if all(s.pot_location == "stove" and s.stove_on and s.pot_filled
               for s in history_slice):
            return state._replace(boiling=True)
        else:
            return state


env = Env(
    [
        ToggleFaucet(),
        ToggleStove(),
        MoveToFaucet(),
        MoveToStove(),
        FillPot(),
        OverfillPot(),
        Boil(),
        Spill()
    ],
    # initial state
    State(boiling=False,
          pot_location="table",
          stove_on=False,
          faucet_on=False,
          pot_filled=False,
          water_spilled=False,
          action=None))


# a policy that can be run at any time point and as the right thing
def small_step_policy(history):
    if len(history) == 1:
        return "action", "move_to_faucet"

    state = history[-1]

    if state.pot_location == "faucet" and not state.faucet_on and not state.pot_filled and state.action != "toggle_faucet":
        return "action", "toggle_faucet"

    if state.pot_location == "faucet" and state.faucet_on and state.pot_filled and state.action != "toggle_faucet":
        return "action", "toggle_faucet"

    if state.pot_location == "faucet" and state.pot_filled and not state.faucet_on and state.action != "move_to_stove":
        return "action", "move_to_stove"

    if state.pot_location == "stove" and not state.stove_on and state.action != "toggle_stove":
        return "action", "toggle_stove"


# a policy that is more of a plan and which should be run in between big steps
big_step_plan = [
    ("action", "move_to_faucet"),
    ("action", "toggle_faucet"),
    None,  # wait for the faucet to fill the pot
    # note that if you put this extra None here, we will keep on waiting after the pot is filled, which causes it to spill
    # boiling still works but the final state has water_spilled=True
    # None,
    ("action", "toggle_faucet"),
    ("action", "move_to_stove"),
    ("action", "toggle_stove"),
    None  # wait for it to boil
]


def bfs_planner(env, goal_predicate):
    """
    Breadth-first search planner that finds a sequence of actions to reach a goal state.
    
    Args:
        env: The environment to plan in
        goal_predicate: Function that takes a state and returns True if it's a goal state
    
    Returns:
        List of actions to reach the goal, or None if no solution found
    """
    # Actions available to the agent (excluding None which means wait)
    actions = [
        None,  # wait goes first so that it is priority, otherwise it will do random actions to "wait"
        ("action", "move_to_faucet"),
        ("action", "move_to_stove"),
        ("action", "toggle_faucet"),
        ("action", "toggle_stove")
    ]

    # Queue of (env_state, plan_so_far) tuples
    queue = deque([(deepcopy(env), [])])

    # Keep track of visited states to avoid cycles
    visited = set()

    while queue:
        curr_env, plan = queue.popleft()

        # Check if current state satisfies goal
        if goal_predicate(curr_env.state):
            return plan

        # Convert state to hashable form for visited set
        # state_tuple = tuple(curr_env.state)
        # if state_tuple in visited:
        #     continue
        # visited.add(state_tuple)

        # Try each action
        for action in actions:
            next_env = deepcopy(curr_env)
            next_env.big_step(action)

            # Add resulting state to queue with updated plan
            queue.append((next_env, plan + [action]))

    return None  # No solution found


# run the planner
plan = bfs_planner(env, lambda s: s.boiling and not s.water_spilled)
# show the plan
print("\nPlan:")
for action in plan:
    print(action)

big_step_plan = plan


def big_step_policy(history):
    global big_step_plan
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
        print(f"At time {env.t}, action=WAIT")

    state = env.big_step(action, verbose=True)
    if state.boiling:
        print("\nWater is boiling!")
        break

# show the history now that we are done
print("\nHistory:")
for state in env.history:
    print(state)
