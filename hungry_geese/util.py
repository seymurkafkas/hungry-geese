from tqdm import tqdm

# Utility Functions
def copy_deep_q_network(network_from, network_to):
    network_to.set_weights(network_from.get_weights())
    return


def get_decorated_episode_iterator(episode_count):
    return tqdm(range(1, episode_count + 1), ascii=True, unit="episodes")
