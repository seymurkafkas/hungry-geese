from hungry_geese.agent import make_agent
from kaggle_environments.envs.hungry_geese.hungry_geese import greedy_agent
from kaggle_environments import make

MODEL_FILE_PATH = "models/weights"

if __name__ == "__main__":
    env = make("hungry_geese", debug=True)
    learning_agent = make_agent(MODEL_FILE_PATH)
    env.reset()
    agents_in_game = [learning_agent, greedy_agent, greedy_agent, greedy_agent]
    env.run(agents_in_game)
    print(env.render(mode="ansi", width=700, height=600))
