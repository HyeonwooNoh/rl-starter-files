import torch

import utils

class Agent:
    """An abstraction of the behavior of an agent. The agent is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, env_id, obs_space, model_dir, argmax=False, num_envs=1):
        _, self.preprocess_obss = utils.get_obss_preprocessor(env_id, obs_space, model_dir)
        self.acmodel = utils.load_model(model_dir)
        self.argmax = argmax
        self.num_envs = num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.acmodel.cuda()
        if self.acmodel.recurrent:
            self.memories = torch.zeros(
                self.num_envs, self.acmodel.memory_rnn.hidden_size * 2,
                device=self.device)

    def get_actions(self, obss, prev_actions):
        preprocessed_obss = self.preprocess_obss(obss, device=self.device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                model_out = self.acmodel(
                    preprocessed_obss, prev_actions, self.memories)
                dist, _, self.memories = model_out[:3]
            else:
                model_out = self.acmodel(preprocessed_obss, prev_actions)
                dist, _ = model_out[:2]

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        if torch.cuda.is_available():
            actions = actions.cpu().numpy()

        return actions

    def get_action(self, obs, prev_action):
        return self.get_actions(
            [obs], torch.tensor([prev_action], dtype=torch.int, device=self.device)).item()

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=self.device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
