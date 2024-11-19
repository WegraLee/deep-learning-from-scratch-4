from collections import deque
import random
import numpy as np
import gym


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0][0] if isinstance(x[0], tuple) else x[0] for x in data]) ## (array([ 0.0024583 , -0.00793713,  0.02181692,  0.02300685], dtype=float32), {})와 같이 tuple로 리턴된 경우에 대응하기 위함
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done


env = gym.make('CartPole-v0', render_mode='human')
replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=32)

for episode in range(10):  # 에피소드 10회 수행
    state = env.reset()[0]
    done = False

    while not done:
        action = 0  # 항상 0번째 행동만 수행
        next_state, reward, terminated, truncated, info = env.step(action) # 경험 데이터 획득
        done = terminated | truncated

        replay_buffer.add(state, action, reward, next_state, done)   # 버퍼에 추가
        state = next_state

# 경험 데이터 버퍼로부터 미니배치 생성
state, action, reward, next_state, done = replay_buffer.get_batch()
print(state.shape)       # (32, 4)
print(action.shape)      # (32,)
print(reward.shape)      # (32,)
print(next_state.shape)  # (32, 4)
print(done.shape)        # (32,)
