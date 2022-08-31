class Transition(object):
    def __init__(self, state, action, reward, new_state, is_done):
        self.state = state
        self.action = action
        self.reward = reward
        self.new_state = new_state
        self.is_done = is_done