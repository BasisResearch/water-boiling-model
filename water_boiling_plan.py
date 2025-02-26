import heapq
from copy import deepcopy

from water_boiling import env


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
    # return state.faucet_on
    return state.boiling and not state.water_spilled


def get_possible_actions(state):
    actions = [("action", "move_to_faucet"), ("action", "move_to_stove"),
               ("action", "toggle_faucet"), ("action", "toggle_stove"),
               ("action", "noop")]
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
    return actions


def plan_with_astar(env):
    open_list = []  # Priority queue for A*
    heapq.heappush(open_list, (0 + heuristic(env.state), 0, env.state, [],
                               env.scheduled_events.copy(), list(env.history)))
    visited = set()

    while open_list:
        _, cost, current_state, plan, scheduled_events, history = heapq.heappop(
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

        for action in get_possible_actions(current_state):
            print(f"Trying action: {action} on state: {current_state}")
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
                (new_cost + h, new_cost, sim_env.state, plan + [action],
                 deepcopy(sim_env.scheduled_events), list(sim_env.history)))

    return None  # No valid plan found


# Running the planner
plan = plan_with_astar(env)
if plan:
    print("\nFound a plan:")
    for step, action in enumerate(plan):
        print(f"Step {step + 1}: {action}")
else:
    print("No valid plan found")
