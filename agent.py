import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Agent(torch.nn.Module):
    def __init__(self, policy_net, observation_space, action_space, comm_output_size, comm_input_size, epsilon):
        super().__init__()
        self.policy_net = policy_net
        self.observation_space = observation_space
        self.action_space = action_space
        self.communication = Communication(comm_input_size, comm_output_size)
        self.collected_resources = set()
        self.epsilon = epsilon 

        # Define Q-learning hyper-parameters
        self.gamma = 0.9
        self.alpha = 0.1
        self.Q = np.zeros((6561, 5))  # TODO: Make this dynamic/scalable       

    def forward(self, state):
        return self.policy_net(state)

    def action(self, state):
        with torch.no_grad():
            logits = self.forward(state)
            probs = torch.softmax(logits, dim=-1)

            #TODO: Process the interaction

            # Epsilon-greedy exploration
            if np.random.random() < self.epsilon:  # Exploration
                action = torch.tensor([np.random.choice(self.action_space.n)], dtype=torch.long)
            else:  # Exploitation
                action = torch.argmax(probs, dim=-1)

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

    def update_Q(self, state, action, next_state, reward):
        self.Q[state, action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[state, action])
    
class Communication(nn.Module):
    def __init__(self, input_size, output_size):
        super(Communication, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
 
def train(agents, shared_policy_net, env, num_episodes=100, render_interval=10, max_steps_per_episode=10):

    optimizer = optim.Adam(shared_policy_net.parameters())

    for episode in range(num_episodes):
        state = env.reset()
        states, actions, rewards = [], [], []
        done = [False] * len(agents)

        # Render the environment and print the full state
        if episode % render_interval == 0:
            print(f"\nEpisode {episode}")
            env.render()
            print("Full state:", env.get_full_state())

        i = 0

        # Sample action for each agent
        actions_timestep = []

        while not all(done):
            # Step the environment with the chosen actions
            next_state, rewards_timestep, done, info, collected_resources = env.step(actions_timestep)

            for i, agent in enumerate(agents):
                if not done[i]:
                    agent_state = (state[0], state[1], state[2+i*2], state[3+i*2])
                    agent_state_tensor = torch.tensor(agent_state, dtype=torch.float32).view(1, -1)
                    action = agent.action(agent_state_tensor)
                    actions_timestep.append(action)
                    agent.update_Q(state, action, next_state, rewards_timestep)
                else:
                    actions_timestep.append(None)

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

            # Communicate the message between agents
            Agent.exchange_messages(agents, communication_states)

            # Store state, action, and reward information for training
            states.append(state)
            actions.append(actions_timestep)
            rewards.append(rewards_timestep)

            state = next_state

            # Update step counter
            i += 1

            # Terminal condition
            if i >= max_steps_per_episode:
                print('\nMaximum steps per episode reached!')
                break

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

    # Annouce either termination or success
    if episode >= num_episodes:
        print(f'\n\nMAXIMUM NUMBER OF {num_episodes} EPISODES REACHED!')
    else:
        print('\n\nCONGRATULATIONS! \nResources have been sucessfully collected!')
        print(f'\nIt took {episode} episodes.')
