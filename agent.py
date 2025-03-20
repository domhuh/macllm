import torch as th
import torch.nn as nn
from torch.distributions.normal import Normal
from llm import MA_LLM_hf

class Agent(nn.Module):
    def __init__(self, observation_space, action_space, max_new_tokens, max_context_length, device, llm_device):
        super().__init__()
        # initialize rag
        self.agent_keys = list(observation_space.keys())
        self.num_agents = len(self.agent_keys)
        self.llm = MA_LLM_hf(self.num_agents, max_new_tokens = max_new_tokens, max_context_length = max_context_length, device=llm_device)

        embedding_size = 384
        self.actor_embedding = nn.ModuleList()
        self.actor_mean = nn.ModuleList()
        self.actor_logstd = nn.ParameterList()
        for agent in self.agent_keys:
            self.actor_embedding.append(nn.Sequential(nn.Linear(observation_space[agent], 256), nn.Tanh(),
                                                      nn.Linear(256, embedding_size)))
            self.actor_mean.append(nn.Sequential(nn.Linear(observation_space[agent] + embedding_size, 64), nn.Tanh(),
                                                 nn.Linear(64, 64), nn.Tanh(),
                                                 nn.Linear(64, action_space[agent])))
            self.actor_logstd.append(nn.Parameter(th.zeros(1, action_space[agent])))
        self.critic = nn.Sequential(nn.Linear(sum(observation_space.values()), 64), nn.Tanh(),
                                    nn.Linear(64, 64), nn.Tanh(),
                                    nn.Linear(64, self.num_agents))
        self.device = device
        self.to(device)

    def get_value(self, x):
        x = x.flatten(1)
        return self.critic(x)

    def get_action_and_value(self, x, actions=None, communicate = True):
        action_outputs = []
        log_probs = []
        entropys = []
        system_prompt = []

        embedding = th.stack([self.actor_embedding[i](x[:,i]) for i in range(self.num_agents)], 1)
        system_prompt = self.llm.retrieve(embedding)
        communication = None
        if communicate:
            communication = self.llm.get_communication(x, system_prompt)
        context_embeddings, _ = self.llm.make_decision_embedding(x, communication, system_prompt)

        for i in range(self.num_agents):
            action_mean = self.actor_mean[i](th.cat([x[:,i], context_embeddings[:,i].to(self.device) + embedding[:,i]],-1))
            action_logstd = self.actor_logstd[i].expand_as(action_mean)
            action_std = th.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            action = probs.sample() if actions is None else actions[:,i]
            action_outputs.append(action)
            log_probs.append(probs.log_prob(action).sum(1))
            entropys.append(probs.entropy().sum(1))

        return th.stack(action_outputs, 1), th.stack(log_probs, 1), th.stack(entropys, 1), self.get_value(x), system_prompt, context_embeddings

