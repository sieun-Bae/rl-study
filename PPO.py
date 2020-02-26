#PPO
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 3 	#epoch
T_horizon = 20 #몇 time step동안 data를 모을지	

class PPO(nn.Module):
	def __init__(self):
		super(PPO, self).__init__()
		self.data = []


		self.fc1 = nn.Linear(4, 256)
		self.fc_pi = nn.Linear(256, 2)
		self.fc_v = nn.Linear(256, 1)
		self.optimizer = optim.Adam(self.parameters(), lr=0.0005)

	def pi(self, x, softmax_dim = 0): #policy network
		x = F.relu(self.fc1(x))
		x = self.fc_pi(x)
		prob = F.softmax(x, dim = softmax_dim)
		return prob

	def v(self, x): #value network
		x = F.relu(self.fc1(x))
		v = self.fc_v(x)
		return v

	def put_data(self, transition):
		self.data.append(transition)

	def make_batch(self):
		s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
		for item in self.data:
			s, a, r, s_prime, prob_a, done = item

			s_lst.append(s)
			a_lst.append([a])
			r_lst.append([r])
			s_prime_lst.append(s_prime)
			prob_a_lst.append([prob_a])
			done_mask = 0 if done else 1
			done_lst.append([done_mask])

		s,a,r,s_prime,done_mask,prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
										 torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype = torch.float), \
										 torch.tensor(done_lst, dtype = torch.float), torch.tensor(prob_a_lst)

		self.data = []
		return s,a,r,s_prime,done_mask,prob_a

	def train_net(self): #GAE
		s,a,r,s_prime,done_mask,prob_a = self.make_batch()

		for i in range(K_epoch):
			td_target = r+ gamma * self.v(s_prime) * done_mask
			delta = td_target - self.v(s) #batch 처리 해서 한번에 연산 pytorch는 바로 연산됨
			delta = delta.detach().numpy()

			advantage_lst = []
			advantage = 0.0
			for delta_t in delta[::-1]: #뒤에서부터 recursive하게 연산
				advantage = gamma * lmbda * advantage + delta_t[0]
				advantage_lst.append([advantage])
			advantage_lst.reverse()
			advantage = torch.tensor(advantage_lst, dtype = torch.float)

			pi = self.pi(s, softmax_dim=1)
			pi_a = pi.gather(1, a)
			#a/b == exp(log(a)-log(b))
			ratio = torch.exp(torch.log(pi_a)-torch.log(prob_a)) #처음 시작은 1이지만 pi_a는 계속 업데이트 됨, prob_a는 data 경험 쌓을 때 과거의 action했을 때 확률

			#pi_a 너무 달라지면 안되니까 clamp 해주는 것
			surr1 = ratio*advantage
			surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip)*advantage
			loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(td_target.detach(), self.v(s)) #-policy loss + value loss
			#detach하는 이유는 target은 고정되어야 하기 때문에 상수로 보겠다. gradient update 하지 말아라. gradient는 v에서만 계산됨
			#안전한 sample에 대해서만 update하겠다.

			self.optimizer.zero_grad()
			loss.mean().backward()
			self.optimizer.step()

def main():
	env = gym.make('CartPole-v1')
	model = PPO()
	score = 0.0
	print_interval = 20

	for n_epi in range(10000):
		s = env.reset()
		done = False
		while not done:
			for t in range(T_horizon):
				prob = model.pi(torch.from_numpy(s).float())
				m = Categorical(prob)
				a = m.sample().item()
				s_prime, r, done, info = env.step(a)
				model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
				s = s_prime

				score += r
				if done:
					break

			model.train_net()
		if n_epi%print_interval==0 and n_epi!=0:
			print("# of episode : {}, avg score : {:.1f}".format(n_epi, score/print_interval))
			score = 0.0

	env.close()

if __name__ == '__main__':
	main()