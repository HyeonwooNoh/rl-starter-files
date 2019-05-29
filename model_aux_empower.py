import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from model_aux import ACAuxModel


class ACAuxEmpowerModel(ACAuxModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False,
                 use_prev_action=False, use_manual_memory_size=False,
                 memory_size=64, use_aux_context=False):
        self.empower_hidden_size = 64
        super().__init__(obs_space, action_space, use_memory, use_text,
                         use_prev_action, use_manual_memory_size, memory_size,
                         use_aux_context)

        self.empower_critic = nn.Sequential(
            nn.Linear(self.embedding_size, self.empower_hidden_size),
            nn.Tanh(),
            nn.Linear(self.empower_hidden_size, 1)
        )

    def forward(self, obs, prev_action, memory):
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = memory[:, :self.memory_rnn.hidden_size]
            aux_x = torch.cat([x, hidden], dim=1)
        else:
            aux_x = x
        aux_hidden = self.aux_embed(aux_x)
        aux_logits = F.log_softmax(self.aux_actor(aux_hidden), dim=1)
        aux_dist = Categorical(logits=aux_logits)

        if self.use_prev_action:
            if self.onehot_prev_action is None or \
                    prev_action.size(0) != self.onehot_prev_action.size(0):
                self.onehot_prev_action = torch.zeros(
                    [prev_action.size(0), self.action_space.n],
                    device=prev_action.device)
            self.onehot_prev_action.zero_()
            self.onehot_prev_action.scatter_(
                1, prev_action.unsqueeze(1).long(), 1)
            x = torch.cat([x, self.onehot_prev_action], dim=1)

        if self.use_aux_context:
            x = torch.cat([x, aux_hidden.detach()], dim=1)

        if self.use_memory:
            hidden = (memory[:, :self.memory_rnn.hidden_size], memory[:, self.memory_rnn.hidden_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        x = self.empower_critic(embedding)
        empower_value = x.squeeze(1)

        return dist, value, memory, aux_dist, empower_value
