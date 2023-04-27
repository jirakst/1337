import torch.nn as nn
from gym import spaces
from environment import TheWorld
import agent
from agent import Agent

def main():
    width = 5
    height = 5
    num_agents = 2
    num_resources = 2
    comm_output_size = 3

    env = TheWorld(width, height, num_agents, num_resources)

    # Create a shared policy network
    shared_policy_net = nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, env.action_space.n)
    )

    # TODO: Calculate the inpute size dynamically
    comm_input_size = 2 + 2 * num_resources + 1

    # Create the agents
    agents = [Agent(shared_policy_net, env.observation_space, env.action_space, comm_output_size, comm_input_size, epsilon=0.6) for _ in range(num_agents)]

    # Train the agents
    agent.train(agents, shared_policy_net, env)

if __name__ == "__main__":
    main()
