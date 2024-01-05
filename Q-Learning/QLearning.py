# 
import gym
import numpy as np

env = gym.make("MountainCar-v0", render_mode="human")

LEARNING_RATE = 0.6
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 2000

DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

print(f"action space shape {env.action_space.n}")
print(f"observation of state shape {env.observation_space.shape}")  # shows position and velocity of car

# Exploration settings
epsilon = 0.3  # not a constant, going to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


RENDER_EVERY = 100  # Render every 100 episodes or as needed

for episode in range(EPISODES):
    discrete_state = get_discrete_state(env.reset()[0])
    done = False

    render = episode % RENDER_EVERY == 0  # Check if the episode is a render episode

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()

        if not done:
            current_q = q_table[discrete_state + (action,)]
            max_future_q = np.max(q_table[new_discrete_state])
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0  # Update the Q-value toward the goal
            print("Goal reached!")

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

env.close()
