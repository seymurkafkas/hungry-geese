from hungry_geese.agent import LearningAgent


LEARNING_CONFIG = {
    "DISCOUNT_RATE": 0.99,  # Decay rate of future rewards
    "REPLAY_BUFFER_SIZE": 50000,  # How many previous transitions are to be remembered
    "MIN_REPLAY_BUFFER_SIZE": 2000,  # The agent only learns after the buffer size exceeds this threshold.
    "BUFFER_SUBBATCH_SIZE": 64,  # Size of the sample to be taken from the buffer.
    "EPISODES_BEFORE_NETWORK_UPDATE": 5,  # Update the target model for every N episodes
    "EPISODE_COUNT": 10000,  # Number of Episodes (Concluded Games)
    "EPSILON": 1,  # Exploration-Exploitation Config #
    "EPSILON_DECAY_RATE": 0.99955,
    "EPSILON_MIN": 0.001,
    "EPISODE_BEFORE_MODEL_SAVE": 50,  # How many episodes will be run before the model is saved
}


def train_and_save(learning_configuration, model_file_path):
    learning_agent = LearningAgent(learning_configuration)
    learning_agent.learn_and_save_model(model_file_path)
    return


if __name__ == "__main__":
    train_and_save(LEARNING_CONFIG, "models/weights")
