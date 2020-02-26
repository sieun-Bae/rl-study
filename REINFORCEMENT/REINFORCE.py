#REINFORCE

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Policy(nn.Module): #pytorch에 있는 module
	def __init__(self):
		super(Policy, self).__init__()
		self.data = []
		self.gamma = 0.99

		self.fc1 = nn.Linear(4, 128) #4차원 -> 128차원
		self.fc2 = nn.Linear(128,2)
		self.optimizer = optim.Adam(self.parameters(), lr=0.0005) 
		###learning rate 10배하면 학습 안됨 -> step size 잘 설정해야함###

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.softmax(self.fc2(x), dim=0)
		return x

	def put_data(self, item):
		self.data.append(item)

	def train(self):
		R = 0 #return
		for r, log_prob in self.data[::-1]:
			R = r + R * self.gamma #gamma 계속 곱해야하므로 뒤에서부터 계산
			loss = -log_prob * R
			self.optimizer.zero_grad()
			loss.backward() #backpropagation되면서 gradient 계산됨 autodiff
			self.optimizer.step()
		self.data = []


def main():
	env = gym.make('CartPole-v1') #environment
	pi = Policy()
	avg_t = 0

	for n_epi in range(10000):
		obs = env.reset() #state
		for t in range(600):
			obs = torch.tensor(obs, dtype=torch.float) 
			#tensor([0.0255, -0.0159, -0.0489, -0.0408]) 각속도 등
			#원래 numpy array인데 torch에 tensor로 들어가야 input 가능
			out = pi(obs)
			#stochastic sampling
			m = Categorical(out) #확률분포 model
			action = m.sample() #tensor(0) action이 하나 나옴
			obs, r, done, info = env.step(action.item()) #state transition
			#action이 tensor이므로 item화하여 scalar 넘겨줌

			pi.put_data((r, torch.log(out[action]))) #logpiceta(s,a)
			#episode가 끝나야 학습할 수 있으므로 일단 저장만 해두는 것
			if done:
				break
		avg_t += t

		pi.train()
		
		if n_epi%20==0 and n_epi!=0:
			print('# of episode :{}, Avg timestep : {}'.format(n_epi, avg_t/20.0))
			avg_t = 0
	
	env.close()

if __name__ == "__main__":
	main()
	

