# Hungry Geese (Kaggle)

This repo uses a model-free, value-based reinforcement learning technique (DQN) to train an agent for the Kaggle competition Hungry Geese.

## Installation

Install the requirements:

```sh
pip install -r requirements.txt
```

Install the hungry_geese module in editable mode:

```sh
pip install -e .
```

## Training a Model

To train and save a network: Run the following command:

```
python train.py
```

Before training, you must declare the learning parameters, with the following schema:

```
    DISCOUNT_RATE                   Decay rate of future rewards
    REPLAY_BUFFER_SIZE              How many previous transitions are to be remembered
    MIN_REPLAY_BUFFER_SIZE          The agent only learns after the buffer size exceeds this threshold.
    BUFFER_SUBBATCH_SIZE            Size of the sample to be taken from the buffer at each training step.
    EPISODES_BEFORE_NETWORK_UPDATE  Update the target model for every N episodes
    EPISODE_COUNT                   The number of episodes (concluded games) for which the agent will be trained
    EPSILON                         Exploration-Exploitation config 
    EPSILON_DECAY_RATE
    EPSILON_MIN
```


## Testing the Model

To run the agent in an exemplary testing environment, simply run

```
 python evaluate.py
```

In this file, the file path of the trained model is supplied as a parameter to the ```DecisionMaker```  constructor.


## Description 
The function approximator for the (state, action) utility values is a neural network composed of convolutional and fully connected layers. The observation received by the agent is used to generate the current state of the game. Afterwards, the features of the state are generated accordingly. During training, the agent chooses to explore or exploit, and it uses the following relationship to update the state-action utilities after each transition.


### State-Action Utilities
Q(s,a) = r + discount_rate * max_A(Q(s', A)),

where (s,a,r,s') denote the former state, the taken action, the received step reward and the transitioned state respectively.
max_A(Q(s, A)) denotes the greatest state-action utility for a given state.


### Exploration-Exploitation
The agent attempts to balance exploration-exploitation with the Epsilon-Greedy Algorithm. The agent retains a buffer of previous transitions, and after each transition, it updates this buffer and samples a collection of transitions. Since the neural network uses a variant of stochastic gradient descent, it must learn from the same transition on multiple instances, to assure convergence. Also, the samples are taken randomly, as the order in which the network is trained highly affects the weight outcomes.  The agent then uses these transitions to update the weights of the neural network, according to given relationship above.


### State Features
The features of the game state are as follows:
    1. Our agent's head position
    2. Adversarial agents' head positions
    3. Our previous head position
    4. Adversarial agents' previous head position
    5. Our tail's position
    6. Adversarial agents' tail positions
    7. Our body positions
    8. Adversarial agents' body positions
    9. Food positions

The world is a grid of size 7x11 ; then, the gamestate is a matrix of size 17x7x11.


## Serialization:

According to Kaggle's submission rules, only a single python file can be submitted, which must include a function definition that receives an observation. Since accessing the pretrained model requires access to external files, the pretrained model can be serialized and hardcoded into the source file as a bytestring. 


The model's weight parameters are serialized with model-serialize.py 
To load, serialize and compress a pre-trained model, use the functions declared in the serializer subpackage.


## Agent Architecture:

The agent is composed of two layers. It perceives at every step of the game, and maintains an internal observation sequence. It asks a DecisionMaker, which uses a neural network internally, to determine the values of each action for each state. Afterwards, it determines if the best suggested action is suicidal. If so, it ignores this action, and applies the second best.