import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
import gym

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False,
                 use_prev_action=False, use_manual_memory_size=False,
                 memory_size=64):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.use_prev_action = use_prev_action
        self.use_manual_memory_size = use_manual_memory_size
        self._semi_memory_size = memory_size
        self.action_space = action_space

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.context_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Define one hot for previous action
        self.onehot_prev_action = None

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        if isinstance(action_space, gym.spaces.Discrete):
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, action_space.n)
            )
        else:
            raise ValueError("Unknown action space: " + str(action_space))

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        if self.use_manual_memory_size:
            return self._semi_memory_size
        else:
            return self.context_size

    @property
    def context_size(self):
        context_size = self.image_embedding_size
        if self.use_prev_action:
            context_size += self.action_space.n
        return context_size

    def forward(self, obs, prev_action, memory):
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

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

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

    def cuda(self, device=None):
        if self.onehot_prev_action is not None:
            self.onehot_prev_action = self.onehot_prev_action.cuda(device=device)
        return super().cuda(device=device)

    def cpu(self):
        if self.onehot_prev_action is not None:
            self.onehot_prev_action = self.onehot_prev_action.cpu()
        return super().cpu()
