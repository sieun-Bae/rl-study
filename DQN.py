#DQN

import gym
import collections #replay buffer에서 쓰일 deque를 위해
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer():
	def __init__(self):
		self.buffer = collections.deque()
		self.batch_size = 32
		self.size_limit = 50000 #buffer의 최대 크기

	def put(self, data):
		self.buffer.append(data)
		if len(self.buffer) > self.size_limit:
			self.buffer.popleft()

	def sample(self, n):
		return random.sample(self.buffer, n)

	def size(self):
		return len(self.buffer)

class Qnet(nn.Module): #left or right possibility
	def __init__(self):
		super(Qnet, self).__init__()
		self.fc1 = nn.Linear(4, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, 2)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def sample_action(self, obs, epsilon):
		out = self.forward(obs)
		coin = random.random()
		if coin < epsilon:
			return random.randint(0,1)
		else:
			return out.argmax().item()

def train(q, q_target, memory, gamma, optimizer, batch_size):
	for i in range(10): #한 episode 끝날때마다 train 호출해서 그냥 10번씩 돌림
		batch = memory.sample(batch_size)
		s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [],[],[],[],[]

		for transition in batch: #transition tuple
			s, a, r, s_prime, done_mask = transition
			s_lst.append(s)
			a_lst.append([a])
			r_lst.append([r]) #s랑 dimension 맞춰주려고 array
			s_prime_lst.append(s_prime) #다음 state
			done_mask_lst.append([done_mask])
		#mini batch 구성하는 과정, 모두 tensor로 바꿈
		s,a,r,s_prime,done_mask = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
								  torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype = torch.float), \
								  torch.tensor(done_mask_lst)
		q_out = q(s) #s는 사실 32개 -> shape: [32,4]
		q_a = q_out.gather(1,a) #실제 한 action의 value [32, 2] e.g. [2,1], [3,2] -> R, L, R, R #1은 1번째에서 골라라
		max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1) #[32, 1] max 취하면 [32], unsqueeze해서 [32,1]
		target = r + gamma * max_q_prime * done_mask
		loss = F.smooth_l1_loss(target, q_a) #-1~1 사이면 제곱에 비례

		optimizer.zero_grad()
		loss.backward()
		optimizer.step() #episode 끝나면 sample 320개 사용된 것

def main():
	env = gym.make('CartPole-v1')
	q = Qnet()
	q_target = Qnet()
	q_target.load_state_dict(q.state_dict()) #q를 q_target으로 복붙
	memory = ReplayBuffer()

	avg_t = 0
	gamma = 0.98
	batch_size = 32
	optimizer = optim.Adam(q.parameters(), lr=0.0005) #q-target은 업데이트하지 않음

	render = False

	for n_epi in range(10000):
		epsilon = max(0.01, 0.08-0.01*(n_epi/200)) #Linear annealing from 8% to 1%
		s = env.reset()

		for t in range(600):
			a = q.sample_action(torch.from_numpy(s).float(), epsilon)
			s_prime, r, done, info = env.step(a)
			done_mask = 0.0 if done else 1.0 #game 끝나면 이후 value 곱해지면 안되므로 0, 안끝나면 1
			memory.put((s,a,r/200.0, s_prime, done_mask))
			s = s_prime

			if render:
				env.render

			if done:
				break
		avg_t += t
		if avg_t/20.0 > 450:
			render = True

		if memory.size() > 2000: #전에는 쌓기만 함
			train(q, q_target, memory, gamma, optimizer, batch_size)
		
		if n_epi%20==0 and n_epi!=0:
			q_target.load_state_dict(q.state_dict()) #target을 20번마다 update
			print("# of episode :{}, Avg timestep : {:.1f}, buffer size : {}. epsilon : {:.1f}%".format(
				n_epi, avg_t/20.0, memory.size(), epsilon*100))

	env.close()

if __name__ == "__main__":
	main()