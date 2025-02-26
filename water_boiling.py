"""An example of a world model that an agent may potentially learn.

It allows the agent to generate a sequence of imaginary states given at sequence
of imaginary action.
The effects of actions such as ToggleFaucet or MoveToFaucet happens after a
delay after their conditions are met. This is also true for processes such as
FillPot.
For processes such as OverfillPot or Boil, the effects happen immediately after
the condition holds for a certain number of time steps.0
"""
from typing import Dict, List
import random

from water_boiling_processes import (AbstractState, CausalProcess,
                                     ToggleFaucet, ToggleStove, MoveToFaucet,
                                     MoveToStove, Noop, FillPot, OverfillPot,
                                     Boil)

random.seed(0)


# The environment keeps track of the state, history, and cause-effects whose condition has been triggered but whose effect has not yet "landed", meaning they are scheduled for the future
class ProcessWorldModel:

    def __init__(self, processes: List[CausalProcess],
                 initial_state: AbstractState) -> None:
        self.processes = processes
        self.state = initial_state
        self.history = [initial_state]
        self.scheduled_events: Dict[int, List[CausalProcess]] = {}
        self.t: int = 0

    def small_step(self, action=None) -> None:
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
                print(f"At time {self.t}, {process.name}.effect scheduled for "
                      f"{schedule_time}")
                if schedule_time not in self.scheduled_events:
                    self.scheduled_events[schedule_time] = list()
                self.scheduled_events[schedule_time].append(process)

        # 4. Check if state changed -- for printing and deactivating the wait
        # process
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

    def big_step(self, action=None) -> AbstractState:
        """This action variable doesn't hold the same value as the action 
        attribute in the state.
        `action` here is passed to small_step for the first time step, and then
        set to None. This is just to initiate the action process.
        In contrast, the action attribute in state is set to that action until 
        the effect take place.
        The loop stops when self.small_step changes the state when action is not
        None (i.e. after the first iteration).
        """
        initial_state = self.state
        while self.state == initial_state:
            self.small_step(action)
            # hypothesis: the action is only non-None for the first time step
            print(f"action in big_step: {action}")
            if action is not None:
                # update the initial state and set action to None
                initial_state = self.state
                action = None
        print(f"Stopping big step because state changed from this:\n"
              f"\t{initial_state}\nto this:\n\t{self.state}")
        return self.state


world_model = ProcessWorldModel(
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
    AbstractState(boiling=False,
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

        if state.pot_location == "faucet" and state.faucet_on and \
            state.pot_filled and state.action != "toggle_faucet":
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
        action = big_step_policy(world_model.history)
        print("\n" * 3)
        if action is not None:
            print(f"At time {world_model.t}, {action[0]}={action[1]}")
        else:
            print(f"At time {world_model.t}, action=NONE")

        state = world_model.big_step(action)
        if state.boiling:
            print("\nWater is boiling!")
            break
    # show the history now that we are done
    print("\nHistory:")
    for state in world_model.history:
        print(state)
