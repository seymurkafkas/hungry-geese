import numpy as np
import math
from kaggle_environments.envs.hungry_geese.hungry_geese import Action
from tensorflow import keras
from hungry_geese.features import make_state


class UtilityApproximator:
    def __init__(self, model_file_path):
        self.model = keras.models.load_model(model_file_path)  # Create model here
        self.model.summary()
        return

    def get_q_values(self, game_state):
        reshaped_game_state = np.expand_dims(game_state, axis=0)
        return self.model.predict(reshaped_game_state)[0]


class DecisionMaker:
    def __init__(self, model_file_path):
        self.action_set = list(Action)
        self.utility_approximator = UtilityApproximator(model_file_path)
        self.previous_action = None
        self.observation_sequence = []
        return

    def choose_action(self, observation):
        q_values = self.get_action_utilities(observation)
        index_of_best_action = np.argmax(q_values)
        if self.is_action_suicidal(self.action_set[index_of_best_action].name):
            q_values[index_of_best_action] = -math.inf
            index_of_best_action = np.argmax(q_values)
        chosen_action = self.action_set[index_of_best_action].name
        self.register_taken_action(chosen_action)
        return chosen_action

    def get_action_utilities(self, observation):
        self.update_observation_sequence(observation)
        current_state = self.make_state()
        q_values = self.utility_approximator.get_q_values(current_state)
        return q_values

    def update_observation_sequence(self, observation):
        if len(self.observation_sequence) == 2:
            self.observation_sequence[0] = self.observation_sequence[1]  # CHANGE
            self.observation_sequence[1] = observation
        else:
            self.observation_sequence.append(observation)
        return

    def is_action_suicidal(self, action):
        return action == self.previous_action

    def register_taken_action(self, action_taken):
        self.previous_action = action_taken

    def get_action_name(self, index):
        return self.action_set[index].name

    def make_state(self):
        return make_state(self.observation_sequence)


def make_agent(model_file_path):
    decision_maker = DecisionMaker(model_file_path)

    def agent(observation):
        chosen_action = decision_maker.choose_action(observation)
        return chosen_action

    return agent
