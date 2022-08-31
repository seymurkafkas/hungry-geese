import numpy as np


def normalize_state_around_head(raw_game_state):
    dy_head, dx_head = np.where(raw_game_state[0])  # Get delta of head
    if len(dy_head) != 0 and len(dx_head) != 0:
        y_dimension_order_array = (np.arange(0, 7) + dy_head[0] - 3) % 7
        x_dimension_order_array = (np.arange(0, 11) + dx_head[0] - 5) % 11
        raw_game_state = raw_game_state[:, y_dimension_order_array, :]
        raw_game_state = raw_game_state[:, :, x_dimension_order_array]
    return raw_game_state


def make_state(observation_sequence):

    # 17 Dimensions Define the Feature Space
    output_feature_matrix = np.zeros((17, 7 * 11), dtype=np.float32)
    last_observation = observation_sequence[-1]
    my_geese_index = last_observation["index"]
    for geese_index, position_array in enumerate(last_observation["geese"]):
        iterated_geese_relative_index = (geese_index - my_geese_index) % 4
        # Dimensions 0-3 store head locations for each agent
        for position in position_array[:1]:
            output_feature_matrix[0 + iterated_geese_relative_index, position] = 1
        # Dimensions 4-7 store tail locations for each agent
        for position in position_array[-1:]:
            output_feature_matrix[4 + iterated_geese_relative_index, position] = 1
        # Dimensions 8-11 store all body locations for each agent
        for position in position_array:
            output_feature_matrix[8 + iterated_geese_relative_index, position] = 1

        # Dimensions 12-15 store previous head locations for each agent, if it exists
    if len(observation_sequence) > 1:
        previous_observation = observation_sequence[-2]
        my_geese_previous_index = previous_observation["index"]
        for geese_index, position_array in enumerate(previous_observation["geese"]):
            iterated_geese_relative_index = (geese_index - my_geese_previous_index) % 4
            for position in position_array[:1]:
                output_feature_matrix[12 + iterated_geese_relative_index, position] = 1

    # Mark all food on dimension 16
    for position in last_observation["food"]:
        output_feature_matrix[16, position] = 1

    raw_feature_matrix = output_feature_matrix.reshape(-1, 7, 11)
    normalized_feature_matrix = normalize_state_around_head(raw_feature_matrix)
    return normalized_feature_matrix
