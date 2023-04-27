import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces


class Agent(torch.nn.Module):
    def __init__(self, policy_net, observation_space, action_space, comm_output_size, comm_input_size):
        super().__init__()
        self.policy_net = policy_net
        self.observation_space = observation_space
        self.action_space = action_space
        self.communication = Communication(comm_input_size, comm_output_size)
        self.collected_resources = set()

    def forward(self, state):
        return self.policy_net(state)

    def action(self, state):
        with torch.no_grad():
            logits = self.forward(state)
            probs = torch.softmax(logits, dim=-1)

            #TODO: Process the interaction

            action = torch.multinomial(probs, num_samples=1)
        return action
    
    def communicate(self, communication_state):
        communication_state_tensor = torch.tensor(communication_state, dtype=torch.float32).view(1, -1)
        message = self.communication(communication_state_tensor)
        return message
    
    def exchange_messages(agents, communication_states):
        messages = [agent.communicate(comm_state) for agent, comm_state in zip(agents, communication_states)]
        for agent, message in zip(agents, messages):
            agent.receive_message(message)

    def receive_message(self, message):
        collected_resource = torch.argmax(message).item()
        self.collected_resources.add(collected_resource)
    

class Communication(nn.Module):
    def __init__(self, input_size, output_size):
        super(Communication, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

 
def train(agents, shared_policy_net, env, num_episodes=1000, render_interval=1):

    optimizer = optim.Adam(shared_policy_net.parameters())

    for episode in range(num_episodes):
        state = env.reset()

        states, actions, rewards = [], [], []
        done = [False] * len(agents)

        # Render the environment and print the full state
        if episode % render_interval == 0:
            print(f"Episode {episode}")
            env.render()
            print("Full state:", env.get_full_state())

        i = 0

        # Iterate until all agents meet the objective
        while not all(done):
            print(f'Iteration {i}:')
            # Sample action for each agent
            actions_timestep = []
            actions_timestep = []
            for i, agent in enumerate(agents):
                if not done[i]:
                    agent_state = (state[0], state[1], state[2+i*2], state[3+i*2])
                    agent_state_tensor = torch.tensor(agent_state, dtype=torch.float32).view(1, -1)
                    action = agent.action(agent_state_tensor)
                    actions_timestep.append(action)
                else:
                    actions_timestep.append(None)

            print(f'Episode {episode}: Actions: {actions_timestep}')

            # Step the environment with the chosen actions
            next_state, rewards_timestep, done, info, collected_resources = env.step(actions_timestep)

            # Announce collected resources
            for agent_idx, collected in enumerate(collected_resources):
                if collected:
                    print(f'Agent {agent_idx} collected a resource!')

            # Get agent/resource position
            communication_states = []
            for agent_idx, agent in enumerate(agents):
                agent_position = next_state[agent_idx * 2: (agent_idx * 2) + 2]
                resources_positions = [next_state[i:i + 2] for i in range(2 * len(agents), len(next_state), 2)]
                communication_state = list(agent_position) + [pos for resource_position in resources_positions for pos in resource_position] + [collected_resources[agent_idx]]  # TODO: Revise this harakiri
                communication_states.append(list(communication_state))

            print(f'Episode {episode}: Communication states: {communication_states}')

            # Communicate the message between agents
            Agent.exchange_messages(agents, communication_states)

            print(f'Episode {episode}: Rewards: {rewards_timestep}, Done: {done}, Info: {info}')
            print(f"State: {state}")
            print(f"Next State: {next_state}")

            print(f'Done values: {done}') 

            # Store state, action, and reward information for training
            states.append(state)
            actions.append(actions_timestep)
            rewards.append(rewards_timestep)

            state = next_state

            print(f'Episode {episode}: Update the state done')

            i += 1

        episode += 1

        # Compute reward-to-go
        num_agents = len(agents)
        rewards_to_go = [[0] * num_agents for _ in range(len(rewards))]
        for i in range(len(rewards) - 1, -1, -1):
            for agent_idx in range(num_agents):
                if i == len(rewards) - 1:
                    rewards_to_go[i][agent_idx] = rewards[i][agent_idx]
                else:
                    rewards_to_go[i][agent_idx] = rewards[i][agent_idx] + rewards_to_go[i + 1][agent_idx]

        print('Compute the reward done')

        # Compute the loss and update the policy network
        optimizer.zero_grad()
        loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

        for t in range(len(states)):
            for agent_idx, agent in enumerate(agents):
                if not done[agent_idx]:  # Only compute loss for non-finished agents
                    agent_state = (states[t][0], states[t][1], states[t][2+agent_idx*2], states[t][3+agent_idx*2])
                    agent_state_tensor = torch.tensor(agent_state, dtype=torch.float32).view(1, -1)
                    logits = agent(agent_state_tensor)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    loss -= torch.gather(log_probs, 1, actions[t][agent_idx].view(1, 1)) * rewards_to_go[t][agent_idx]

        loss.backward()
        optimizer.step()
