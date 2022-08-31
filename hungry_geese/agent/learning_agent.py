import numpy as np
import random
from hungry_geese.util import copy_deep_q_network, get_decorated_episode_iterator
from hungry_geese.memory import ReplayBuffer, Transition
from hungry_geese.features import make_state, ACTION_SET
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Activation, Dense, MaxPooling2D, Flatten
from kaggle_environments.envs.hungry_geese.hungry_geese import greedy_agent
from kaggle_environments import make


class LearningAgent(object):
    def __init__(self, learning_configuration):
        self.model = create_deep_q_model()  # Create model here
        self.target_model = create_deep_q_model()  # Create target model
        copy_deep_q_network(self.model, self.target_model)
        self.replay_buffer = ReplayBuffer(learning_configuration)
        self.epsilon = 1
        self.episode_count = learning_configuration["EPISODE_COUNT"]
        self.epsilong_decay_rate = learning_configuration["EPSILON_DECAY_RATE"]
        self.epsilon_minimum = learning_configuration["EPSILON_MIN"]
        self.discount_rate = learning_configuration["DISCOUNT_RATE"]
        self.update_target_after_x_episodes = learning_configuration[
            "EPISODES_BEFORE_NETWORK_UPDATE"
        ]
        self.remaining_episodes_before_target_update = (
            self.update_target_after_x_episodes
        )
        self.observation_sequence = []
        self.episode_per_save = learning_configuration["EPISODE_BEFORE_MODEL_SAVE"]
        return

    def should_exploit(self):
        return np.random.random() > self.epsilon

    def update_epsilon(self):
        if self.epsilon > self.epsilon_minimum:
            self.epsilon *= self.epsilong_decay_rate
        return

    def get_q_values(self, game_state):
        reshaped_game_state = np.expand_dims(game_state, axis=0)
        return self.model.predict(reshaped_game_state)[0]

    def get_target_q_values(self, game_state):
        return self.target_model.predict(game_state)  # stub

    def choose_action(self, game_state):
        if self.should_exploit():
            chosen_action = np.argmax(self.get_q_values(game_state))
        else:
            chosen_action = random.randrange(len(ACTION_SET))
        return chosen_action

    def save_transition_to_buffer(self, transition):
        self.replay_buffer.save_transition(transition)
        return

    def update_observation_sequence(self, observation):
        if len(self.observation_sequence) == 2:
            self.observation_sequence[0] = self.observation_sequence[1]  # CHANGE
            self.observation_sequence[1] = observation
        else:
            self.observation_sequence.append(observation)
        return

    def reset_observation_sequence(self):
        self.observation_sequence = []
        return

    def make_state(self):
        return make_state(self.observation_sequence)

    def should_update_target_network(self, is_episode_over):
        if is_episode_over:
            self.remaining_episodes_before_target_update -= 1
            if self.remaining_episodes_before_target_update == 0:
                self.remaining_episodes_before_target_update = (
                    self.update_target_after_x_episodes
                )  # reset the update counter
                return True
        return False

    def update_target_network(self):
        copy_deep_q_network(self.model, self.target_model)
        return

    def train(self, isDone):
        if not self.replay_buffer.has_sufficient_elements_for_training():
            return
        random_transition_sample = self.replay_buffer.get_random_sample()
        current_state_array = np.array(
            [transition.state for transition in random_transition_sample]
        )
        new_state_array = np.array(
            [transition.new_state for transition in random_transition_sample]
        )
        current_state_q_values_array = self.model.predict(current_state_array)
        new_state_q_values_array = self.target_model.predict(new_state_array)
        # Input-Output for Supervised Learning
        for index, transition in enumerate(random_transition_sample):
            if transition.is_done:
                q_value = transition.reward
            else:
                max_q_value_for_new_state = np.max(new_state_q_values_array[index])
                q_value = (
                    transition.reward + self.discount_rate * max_q_value_for_new_state
                )
            current_state_q_values_array[index][transition.action] = q_value
        self.model.fit(
            np.array(current_state_array),
            np.array(current_state_q_values_array),
            batch_size=self.replay_buffer.random_sample_size,
            verbose=0,
            shuffle=False,
        )
        if self.should_update_target_network(isDone):
            self.update_target_network()
        return

    def learn_and_save_model(self, model_file_path):
        print("hey")
        env = make("hungry_geese", debug=False)
        agents_in_game = [None, greedy_agent, greedy_agent, greedy_agent]
        trainer = env.train(agents_in_game)
        training_episodes = get_decorated_episode_iterator(self.episode_count)
        for episode in training_episodes:
            current_step = 0
            current_episode_reward = 0
            initial_observation = trainer.reset()
            self.reset_observation_sequence()
            self.update_observation_sequence(initial_observation)
            current_state = self.make_state()
            is_episode_over = False
            while not is_episode_over:
                current_step += 1
                decided_action = self.choose_action(current_state)
                observation, reward, is_done, info = trainer.step(
                    ACTION_SET[decided_action].name
                )
                self.update_observation_sequence(observation)
                new_state = self.make_state()
                transition = Transition(
                    current_state, decided_action, reward, new_state, is_done
                )
                # print(env.render(mode="ansi"))
                current_episode_reward += reward
                self.save_transition_to_buffer(transition)
                self.train(is_done)
                current_state = new_state
                is_episode_over = is_done
            average_reward = current_episode_reward / current_step
            self.update_epsilon()
            print("Current Reward:", current_episode_reward)
            if episode % self.episode_per_save == 0:
                self.save_model_to_file(model_file_path)
        return

    def load_model_from_file(self, model_file_path):
        return

    def save_model_to_file(self, model_file_path):
        self.model.save(model_file_path)
        return


# Utility
def create_deep_q_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding="same", input_shape=(17, 7, 11)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(padding="same", pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding="same"))  # 1nd convolution layer
    model.add(Activation("relu"))
    model.add(MaxPooling2D(padding="same", pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), padding="same"))  # 2nd convolution layer
    model.add(Activation("relu"))
    model.add(MaxPooling2D(padding="same", pool_size=(2, 2)))
    model.add(Flatten())  # Transform feature map into 1D vector for the next layer
    model.add(Dense(64))
    model.add(
        Dense(len(ACTION_SET), activation="linear")
    )  # The number of actions available
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
    return model
