import random
class WarehouseAgentReflex:
    def __init__(self):
        pass
    def act(self, state):
        # Unpack state using correct keys from WarehouseEnv
        pos = state['robot_pos']
        carrying = state['has_item']
        pickup = state['pickup_pos']
        dropoff = state['dropoff_pos']
        # Valid actions are always ['N', 'E', 'S', 'W', 'WAIT', 'PICK', 'DROP']
        # We'll filter for valid moves by checking the environment, but here we assume all are possible
        valid_actions = []
        # Only allow moves that don't hit walls, and only allow PICK/DROP when appropriate
        # For simplicity, assume all actions are valid (as in the original agent), or adapt if env provides valid_actions
        if 'valid_actions' in state:
            valid_actions = state['valid_actions']
        else:
            valid_actions = ['N', 'E', 'S', 'W', 'WAIT', 'PICK', 'DROP']
        # Rule 1: At pickup and not carrying → PICK
        if pos == pickup and not carrying and 'PICK' in valid_actions:
            return 'PICK'
        # Rule 2: At dropoff and carrying → DROP
        if pos == dropoff and carrying and 'DROP' in valid_actions:
            return 'DROP'
        # Rule 3: Carrying, need to move toward dropoff
        if carrying and dropoff:
            if dropoff[0] < pos[0] and 'N' in valid_actions:
                return 'N'
            if dropoff[0] > pos[0] and 'S' in valid_actions:
                return 'S'
            if dropoff[1] < pos[1] and 'W' in valid_actions:
                return 'W'
            if dropoff[1] > pos[1] and 'E' in valid_actions:
                return 'E'
        # Rule 4: Not carrying, need to move toward pickup
        elif not carrying and pickup:
            if pickup[0] < pos[0] and 'N' in valid_actions:
                return 'N'
            if pickup[0] > pos[0] and 'S' in valid_actions:
                return 'S'
            if pickup[1] < pos[1] and 'W' in valid_actions:
                return 'W'
            if pickup[1] > pos[1] and 'E' in valid_actions:
                return 'E'
        # Fallback: random valid action
        return random.choice(valid_actions)