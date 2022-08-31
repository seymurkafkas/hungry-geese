import random


class ReplayBuffer(object):
    def __init__(self, learning_configuration):
        self.max_size = learning_configuration["REPLAY_BUFFER_SIZE"]
        self.random_sample_size = learning_configuration["BUFFER_SUBBATCH_SIZE"]
        self.min_size_to_learn = learning_configuration["MIN_REPLAY_BUFFER_SIZE"]
        self.transition_list = []
        self.index_of_least_recent_transition = 0
        return

    def get_current_buffer_size(self):
        return len(self.transition_list)

    def has_sufficient_elements_for_training(self):
        return len(self.transition_list) >= self.min_size_to_learn

    def save_transition(self, transition):
        if len(self.transition_list) >= self.max_size:
            index = self.index_of_least_recent_transition
            self.transition_list[index] = transition
            self.index_of_least_recent_transition = (index + 1) % self.max_size
        else:
            self.transition_list.append(transition)
        return

    def get_random_sample(self):
        return random.sample(self.transition_list, self.random_sample_size)
