import gym
from gym import spaces
import numpy as np
import random

class ConnectFourEnvironment(gym.Env):
    def __init__(self):
        self.rows = 6
        self.columns = 7
        self.board = np.zeros((self.rows, self.columns))

        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.rows, self.columns), dtype=int)

    def reset(self):
        self.board = np.zeros((self.rows, self.columns))
        return self.board

    def is_valid_move(self, action):
        return self.board[0, action] == 0

    def step(self, action):
        if not self.is_valid_move(action):
            return self.board, -1, True, {}  # invalid move
        for r in range(self.rows - 1, -1, -1):
            if self.board[r, action] == 0:
                self.board[r, action] = 1
                break
        win = self.check_win(1)
        done = win
        reward = 1 if win else 0

        if not done:  # if game is not finished, opponent makes a move
            opponent_action = random.choice([c for c in range(self.columns) if self.is_valid_move(c)])
            for r in range(self.rows - 1, -1, -1):
                if self.board[r, opponent_action] == 0:
                    self.board[r, opponent_action] = 2
                    break
            win = self.check_win(2)
            done = win or len([c for c in range(self.columns) if self.is_valid_move(c)]) == 0
            reward = -1 if win else reward

        return self.board, reward, done, {}

    def check_win(self, player):
        # check horizontal
        for r in range(self.rows):
            for c in range(self.columns - 3):
                if self.board[r, c] == self.board[r, c + 1] == self.board[r, c + 2] == self.board[r, c + 3] == player:
                    return True
        # check vertical
        for r in range(self.rows - 3):
            for c in range(self.columns):
                if self.board[r, c] == self.board[r + 1, c] == self.board[r + 2, c] == self.board[r + 3, c] == player:
                    return True
        # check right diagonal
        for r in range(self.rows - 3):
            for c in range(self.columns - 3):
                if self.board[r, c] == self.board[r + 1, c + 1] == self.board[r + 2, c + 2] == self.board[
                    r + 3, c + 3] == player:
                    return True
        # check left diagonal
        for r in range(3, self.rows):
            for c in range(self.columns - 3):
                if self.board[r, c] == self.board[r - 1, c + 1] == self.board[r - 2, c + 2] == self.board[
                    r - 3, c + 3] == player:
                    return True
        return False

    def render(self, mode='human'):
        print("  ".join(map(str, range(self.columns))))
        print("\n".join(["  ".join(map(str, row)) for row in self.board.astype(int)]))
        print()

class QLearningAgent:
    def __init__(self, actions, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        self.q_table = {}  # state-action pairs

    def get_action(self, state):
        state = tuple(state.flatten())
        if random.uniform(0, 1) < self.epsilon or state not in self.q_table:
            return random.choice(self.actions)  # explore
        else:
            return np.argmax(self.q_table[state])  # exploit

    def update_q_table(self, state, action, reward, next_state):
        state = tuple(state.flatten())
        next_state = tuple(next_state.flatten())
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))

        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))

        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value

def train_agent(agent, env, episodes):
    wins = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
        wins.append(reward)
        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1}, Q-table size: {len(agent.q_table)}")
    return wins

def simulate_game(agent1, agent2, env):
    state = env.reset()
    env.render()
    done = False
    winner = None
    while not done:
        action1 = agent1.get_action(state)
        state, reward1, done, _ = env.step(action1)
        env.render()
        if done:
            winner = 'Agent 1'
            break
        action2 = agent2.get_action(state)
        state, reward2, done, _ = env.step(action2)
        env.render()
        if done:
            winner = 'Agent 2'
    print(f"The winner is: {winner}")
    return env.board


env = ConnectFourEnvironment()
agent = QLearningAgent(actions=list(range(env.action_space.n)))
agent2 = QLearningAgent(actions=list(range(env.action_space.n)))

train_agent(agent2, env, 5000)
train_agent(agent, env, 5000)

simulate_game(agent, agent2, env)
