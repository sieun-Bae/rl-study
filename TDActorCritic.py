#TD Actor Critic

#REINFORCE

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
	def __init__(self):
		super(ActorCritic, self).__init__()
		#representation layer를 공유한다
		self.loss_lst = []

		self.fc1 = nn.Linear(4, 128)
		self.fc_pi = nn.Linear(128, 2)
		self.fc_v = nn.Linear(128, 1)
		self.optimizer = optim.Adam(self.parameters(), lr=0.002)
		### learning rate 커져도 됨 ###

	def forward(self, x):
		x = F.relu(self.fc1(x))
		pol = self.fc_pi(x)
		pi = F.softmax(pol, dim = 0)
		v = self.fc_v(x)
		return pi, v

	def gather_loss(self, loss):
		self.loss_lst.append(loss.unsqueeze(0)) #차원을 늘려준 것

	def train(self):
		loss = torch.cat(self.loss_lst).sum()
		loss = loss/len(self.loss_lst) ### 나눠주니 학습이 더 잘되더라 (loss scaling)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		self.loss_lst = []

def main():
	env = gym.make('CartPole-v1')
	model = ActorCritic()
	gamma = 0.99

	avg_t = 0

	for i_episode in range(10000):
		obs = env.reset()
		#loss_lst = []

		for t in range(600):
			obs = torch.from_numpy(obs).float()
			pi, v = model(obs)
			m = Categorical(pi)
			action = m.sample()

			#env.render()

			obs, r, done, info = env.step(action.item())
			_, next_v = model(torch.from_numpy(obs).float())
			delta = r + gamma * next_v - v
			loss = -torch.log(pi[action]) * delta.item() + delta*delta
			model.gather_loss(loss) #한 episode 모아서 한 것 -> step마다 할 수 있지만 구현해보니 모아서 한 것이 더 나았음

			if done:
				break

		model.train()
		avg_t += t 

		if i_episode%20==0 and i_episode!=0:
			print('# of episode :{}, Avg timestep : {}'.format(i_episode, avg_t/20.0))
			avg_t = 0
	
	env.close()

if __name__ == "__main__":
	main()
	

