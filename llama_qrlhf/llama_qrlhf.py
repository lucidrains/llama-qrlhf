import torch
from torch.nn import Module
from torch.utils.data import Dataset
from torch import nn, einsum, Tensor
import torch.nn.functional as F

from einops import rearrange, repeat

from ema_pytorch import EMA

from beartype import beartype
from beartype.typing import Optional

from torchtyping import TensorType

# helper functions

def exists(v):
    return v is not None

# tensor helpers

def batch_select_indices(t, indices):
    indices = rearrange(indices, '... -> ... 1')
    selected = t.gather(-1, indices)
    return rearrange(selected, '... 1 -> ...')

# Q functions

# autoregressive formulation from https://qtransformer.github.io/
# adapted to language modeling

def autoregressive_q_learn(
    model:          Module,
    ema_model:      Module,
    states:         TensorType['b', 'n', int],     # the entire sequence, containing prompt and generated
    prompt_len:     TensorType['b', int],          # the length of the sequence that is the preceding prompt
    next_states:    TensorType['b', int],          # selected action becomes the next state
    rewards:        TensorType['b', 'n', float],   # the reward could be given at the very end, or interspersed (say the language model made a first correct reasoning then fails later on)
    eos_id:         Optional[int] = None,          # calculate the done from the <eos> token id
    discount_gamma: float = 0.998                  # reward discount factor, encourages brevity of generated answer

) -> TensorType[()]:
    """
    einops

    b - batch
    n - sequence len
    """
    seq_len, device = states.shape[-1], states.device

    # because greek unicode is nice to look at

    γ = discount_gamma

    # get predicted Q for each action

    q_pred_all_actions = model(states)
    q_pred = batch_select_indices(q_pred_all_actions, actions)

    # append next state to current one, for deriving q target

    q_target_input = pack([states[:, 1:], next_state], 'b *')

    # get target Q

    q_target = ema_model(q_target_input)
    q_target = q_target_all_actions.max(dim = -1).values

    # anything after the first done flag will be considered terminal

    if exists(eos_id):
        done = states == eos_id
        dones = dones.cumsum(dim = -1) > 0
        dones = F.pad(dones, (1, -1), value = False)

        not_terminal = (~dones).float()

        # rewards should not be given on and after terminal step

        rewards = rewards * not_terminal
        q_target = q_target.masked_fill(dones, 0.)

    # main contribution of the paper is the following logic
    # section 4.1 - eq. 1

    # where a reward is not given, Q pred for time t is the max(Q target) of t + 1

    losses_without_rewards = F.mse_loss(q_pred, q_target, reduction = 'none')

    # take care of the time steps with rewards given. classic Bellman's

    q_target_with_rewards = rewards + γ * q_target

    losses_with_rewards = F.mse_loss(q_pred, q_target_with_rewards, reduction = 'none')

    # final losses

    losses = torch.where(
        rewards > 0.,
        losses_with_reward,
        losses_without_rewards
    )

    # perform a masked mean
    # as only the 'q logits' starting from the last token of the prompt is considered an 'action'

    is_action_mask = torch.arange(seq_len, device = device) > rearrange(prompt_len - 1, 'b -> b 1')
    losses = losses[is_action_mask]

    return losses.mean()

def conservative_regularization_loss(
    q_values: TensorType['b', 'n', 'a', float],
    states_and_actions: TensorType['b', 'n', int],
    action_mask: TensorType['b', 'n', bool],
    reward_min: float = 0.
) -> TensorType[()]:

    batch, seq_len, num_actions, device = *q_values.shape, q_values.device
    non_dataset_actions = torch.arange(num_actions, device = device) == rearrange(states_and_actions, '... -> ... 1')

    q_values = q_values[~non_dataset_actions]
    q_values = rearrange(q_values, '(b n a) -> b n a', b = batch, n = seq_len)
    q_values = q_values[action_mask]

    reward_min = torch.full((), reward_min, device = device) * seq_len

    return F.mse_loss(q_values, reward_min)

# main classes

class QRLHF(Module):
    @beartype
    def __init__(
        self,
        model: Module,
        dataset: Dataset,
        ema_kwargs: dict = dict(
            beta = 0.99
        )
    ):
        super().__init__()

        self.lm = model
        self.lm_target = EMA(model, **ema_kwargs)

    def forward(self):
        raise NotImplementedError
