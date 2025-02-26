import heapq
from copy import deepcopy
import itertools  # Add this import for the counter

from water_boiling import world_model
from water_boiling_processes import (Noop, ToggleFaucet, ToggleStove, 
                                    MoveToFaucet, MoveToStove)


def heuristic(state):
    h = 0
    # if not state.pot_filled:
    #     h += 10
    # if state.pot_location != "stove":
    #     h += 5
    # if not state.stove_on:
    #     h += 5
    # if state.water_spilled:
    #     h += 20  # Penalty for undesired outcomes
    # if state.boiling:
    #     h = 0
    return h


def is_goal(state):
    # return state.pot_location == "faucet"
    # return state.faucet_on and state.pot_location == "faucet"
    # return state.pot_filled
    return state.boiling and not state.water_spilled

action_to_process = {
    "noop": Noop(),
    "toggle_faucet": ToggleFaucet(),
    "toggle_stove": ToggleStove(),
    "move_to_faucet": MoveToFaucet(),
    "move_to_stove": MoveToStove()
}

def get_possible_actions(history):
    actions = [("action", "noop"),
               ("action", "move_to_faucet"), ("action", "move_to_stove"),
               ("action", "toggle_faucet"), ("action", "toggle_stove"),
               ]
    for action in actions:
        history_copy = deepcopy(history)
        history_copy[-1] = history_copy[-1]._replace(action=action[1])
        if action_to_process[action[1]].condition_at_start(history_copy):
            yield action
    # if state.pot_location != "faucet":
    #     actions.append(("action", "move_to_faucet"))
    # if state.pot_location == "faucet" and not state.faucet_on:
    #     actions.append(("action", "toggle_faucet"))
    # if state.pot_filled and state.faucet_on:
    #     actions.append(("action", "toggle_faucet"))
    # if state.pot_location == "faucet" and state.pot_filled:
    #     actions.append(("action", "move_to_stove"))
    # if state.pot_location == "stove" and not state.stove_on:
    #     actions.append(("action", "toggle_stove"))
    # return actions


def plan_with_astar(env):
    open_list = []  # Priority queue for A*
    counter = itertools.count()
    heapq.heappush(open_list, (0 + heuristic(env.state), 0, next(counter), 
                               env.state, [],
                               env.scheduled_events.copy(), list(env.history)))
    visited = set()

    while open_list:
        _, cost, _, current_state, plan, scheduled_events, history = heapq.heappop(
            open_list)

        if is_goal(current_state):
            return plan  # Return the action plan

        state_signature = (current_state.boiling, current_state.pot_location,
                           current_state.stove_on, current_state.faucet_on,
                           current_state.pot_filled,
                           current_state.water_spilled)

        if state_signature in visited:
            continue
        visited.add(state_signature)

        for action in get_possible_actions(history):
            print(f"\nTrying action: {action} on state: {current_state}, after plan {plan}")
            # Reset environment to the current state
            sim_env = deepcopy(env)
            sim_env.state = deepcopy(current_state)
            sim_env.scheduled_events = deepcopy(scheduled_events)
            sim_env.history = deepcopy(history)
            sim_env.t = len(sim_env.history) - 1

            sim_env.big_step(action)
            new_cost = cost + 1  # Each action has a cost of 1
            h = heuristic(sim_env.state)
            heapq.heappush(
                open_list,
                (new_cost + h, new_cost, next(counter), sim_env.state, plan + [action],
                 deepcopy(sim_env.scheduled_events), list(sim_env.history)))

    return None  # No valid plan found


# Running the planner
plan = plan_with_astar(world_model)
if plan:
    print("\nFound a plan:")
    for step, action in enumerate(plan):
        print(f"Step {step + 1}: {action}")
else:
    print("No valid plan found")

def big_step_policy(history):
    global plan
    del history
    if len(plan) > 0:
        return plan.pop(0)
    else:
        return None

# # Running the plan
# for _ in range(100):
#     action = big_step_policy(world_model.history)
#     print("\n" * 3)
#     if action is not None:
#         print(f"At time {world_model.t}, {action[0]}={action[1]}")
#     else:
#         print(f"At time {world_model.t}, action=NONE")

#     state = world_model.big_step(action)
#     if is_goal(state):
#         print("Goal reached!")
#         break
