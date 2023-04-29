PROJECT DESCRIPTION:

1) A detailed description of the multi-agent system, including the agents' architecture, abilities, and behaviors:

The multi-agent system is designed to simulate an environment (TheWorld) where agents move and collect resources. There are two agents in the environment, which is currently a 3x3 grid. The agents have a shared policy network, which is a simple neural network with one hidden layer. The policy network's purpose is to determine the best action for each agent to take at a given state. The agents follow an epsilon-greedy exploration strategy, choosing a random action with probability epsilon and the action with the highest predicted reward with probability (1 - epsilon).

The agents can take five actions: move up, move down, move left, move right, or attempt to collect a resource at their current position. When an agent collects a resource, it receives a reward of +1. The agent's objective is to collect all resources in the environment. They communicate via a shared memory that is propraged through the policy network.


2) Code implementation of the system, including any machine learning models or algorithms used:

The code is organized into three main parts: the environment (TheWorld), the agent (Agent), and the main script (main.py). The environment is implemented using the gym.Env class, and it provides the basic structure for the agents to interact with the world. The agent class is a subclass of torch.nn.Module, which houses the shared policy network and the agent's action and reward functions.

The training process is implemented in the train function, which uses the Adam optimizer and the ExponentialLR learning rate scheduler. The agents are trained using policy gradient, and the reward-to-go is used to calculate the loss for each step.


3) Documentation of the experiment setup and evaluation metrics used to assess the system's performance:

The agents are currently trained for 300 episodes, and the environment is rendered every 10 episodes to visualize the agents' progress. The maximum number of steps per episode is set to 5, as of now. The agents' performance is evaluated based on the number of resources they collect over a number of episodes and total training steps. The experiment also calculates the Nash equilibria of the given payoff matrix using the NashPy library to determine the optimal strategy for each agent in a game-theoretic sense.


4) Analysis of the results and any recommendations for improving the system:

- Increase the maximum number of steps per episode or remove the limit entirely, allowing the agents more time to explore and collect resources.
- Implement a more sophisticated exploration strategy, such as decaying epsilon over time to encourage more exploitation as the agents learn.
- Use a more complex policy network architecture or other reinforcement learning algorithms (e.g., DQN, PPO) to potentially improve learning efficiency and performance.
- Experiment with different learning rate schedulers, such as StepLR, CosineAnnealingLR, or ReduceLROnPlateau, to further optimize the learning process.
