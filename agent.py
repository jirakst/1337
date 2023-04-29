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
        # self.communication = Communication(comm_input_size, comm_output_size)
        self.collected_resources = set()
        # Define epsilon hyperparameter for e-greedy
        self.epsilon = epsilon  
        # Define the  Q-learning hyperparametrs
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        # self.Q = np.zeros((observation_space, action_space.n))

        # Initialize the Q-matrix
        # self.Q = np.zeros((np.prod(observation_space.shape), action_space.n))   

    def forward(self, state):
        return self.policy_net(state)

    def action(self, state):
        with torch.no_grad():
            logits = self.forward(state)
            probs = torch.softmax(logits, dim=-1)

            # Update the state to match the Q-matrix dimensions
            # state = np.ravel_multi_index(tuple(state), self.observation_space.shape)  # This is actually taken care by the policy net   

            #TODO: Process the interaction

            # Epsilon-greedy exploration
            if np.random.random() < self.epsilon:  # Exploration
                action = torch.tensor([np.random.choice(self.action_space.n)], dtype=torch.long)
            else:  # Exploitation
                action = torch.argmax(probs, dim=-1)

            action = torch.multinomial(probs, num_samples=1)
        return action.item()

    def reward_function(self, collected_resources):
        reward = 0
        for resource_collected in collected_resources:
            if resource_collected:
                reward += 1
        return reward
    '''
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

    # Update the Q-matrix
    def update_Q(self, state, action, reward, next_state, next_action):
        self.Q[state, action] += self.learning_rate * (reward + self.discount_factor * self.Q[next_state, next_action] - self.Q[state, action])

class Communication(nn.Module):
    def __init__(self, input_size, output_size):
        super(Communication, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    '''
 
def train(agents, shared_policy_net, env, num_episodes=5000, render_interval=10, max_steps_per_episode=5):

    # Create the optimizer
    optimizer = optim.Adam(shared_policy_net.parameters())

    total_steps = 0

    dones = [False] * len(agents)

    for episode in range(num_episodes):
        state = env.reset()
        states, actions, rewards = [], [], []
        

        # Render the environment and print the full state
        if episode % render_interval == 0:
            print(f"\nEpisode {episode}")
            env.render()
            print("Full state:", env.get_full_state())

        # Reset the total collected resources
        total_collected_resources = [0] * len(agents)

        steps = 0

        # Iterate until all agents meet the terminal condition
        while not all(dones):  # sum(dones) < len(agents):
            actions_timestep = []
            for i, agent in enumerate(agents):
                if not dones[i]:
                    agent_state = agent_state = (state[0], state[1], state[2+i], state[3+i])
                    agent_state_tensor = torch.tensor(agent_state, dtype=torch.float32).view(1, -1)
                    action = agent.action(agent_state_tensor)
                    actions_timestep.append(action)
                else:
                    actions_timestep.append(None)

            # Step the environment with the chosen actions
            next_state, rewards_timestep, dones, info, collected_resources = env.step(actions_timestep)

            # Compute rewards based on the reward function
            rewards_timestep = [agent.reward_function(collected_resources) for agent in agents]

            # Update the Q-values
            # agent.update_Q(state, action, rewards_timestep, next_state, next_state)

            # Announce collected resources
            for agent_idx, collected in enumerate(collected_resources):
                if collected:
                    total_collected_resources[agent_idx] += collected
                    # print(f'\nAgent {agent_idx} collected a resource!')

            # Update dones list for agents who collected all resources
            for agent_idx, collected_resources_count in enumerate(total_collected_resources):
                if collected_resources_count == len(env.resource_positions):
                    dones[agent_idx] = True

            # Check if all resources have been collected and break the loop  # This is redudant
            if all([collected == len(env.resource_positions) for collected in total_collected_resources]):
                print("All resources have been collected!")
                break
            '''
            # Get agent/resource position
            communication_states = []
            for agent_idx, agent in enumerate(agents):
                agent_position = next_state[agent_idx * 2: (agent_idx * 2) + 2]
                resources_positions = [next_state[i:i + 2] for i in range(2 * len(agents), len(next_state), 2)]
                communication_state = list(agent_position) + [pos for resource_position in resources_positions for pos in resource_position] + [collected_resources[agent_idx]]  # TODO: Revise this harakiri
                communication_states.append(list(communication_state))

            # Communicate the message between agents
            # Agent.exchange_messages(agents, communication_states)
            '''
            # Store state, action, and reward information for training
            states.append(state)
            actions.append(actions_timestep)
            rewards.append(rewards_timestep)

            state = next_state
            
            # Update step counter
            steps += 1
            total_steps += 1

            # Terminal condition
            if steps >= max_steps_per_episode:
                # print('\nMaximum steps per episode reached!')
                break

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
                if not dones[agent_idx]:  # Only compute loss for non-finished agents
                    agent_state = (states[t][0], states[t][1], states[t][2+agent_idx*2], states[t][3+agent_idx*2])
                    agent_state_tensor = torch.tensor(agent_state, dtype=torch.float32).view(1, -1)
                    logits = agent(agent_state_tensor)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    action_tensor = torch.tensor(actions[t][agent_idx], dtype=torch.long).view(1, 1)
                    loss = loss - torch.gather(log_probs, 1, action_tensor) * rewards_to_go[t][agent_idx]

        loss.backward()
        optimizer.step()
        
        # Check if all resources have been collected by the agents
        if all([pos == (-1, -1) for pos in env.resource_positions]):
            dones = [True] * len(agents)
            print(f"\nGlobal state after {episode + 1} episodes and {total_steps} total steps:")
            for agent_idx, collected in enumerate(total_collected_resources):
                print(f"Agent {agent_idx} collected {collected} resources")
            break 

        # Check either for the termination
        if episode >= num_episodes:
            print(f'\n\nMAXIMUM NUMBER OF {num_episodes} EPISODES REACHED!')
            print(f"\nGlobal state after {episode + 1} episodes and {total_steps} total steps:")
            for agent_idx, collected in enumerate(total_collected_resources):
                print(f"Agent {agent_idx} collected {collected} resources")
            break
