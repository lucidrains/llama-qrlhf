import torch
from torch import nn, einsum, Tensor
from torch.nn import Module

from einops import rearrange, repeat

from beartype import beartype

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

) -> Tensor:
    """
    einops

    b - batch
    n - sequence len
    """

    batch, seq_len = states.shape, states.device

    # anything after the first done flag will be considered terminal

    if exists(eos_id):
        done = states == eos_id
        dones = dones.cumsum(dim = -1) > 0
        dones = F.pad(dones, (1, -1), value = False)

        not_terminal = (~dones).float()

        # rewards should not be given on and after terminal step

        rewards = rewards * not_terminal

    # because greek unicode is nice to look at

    γ = self.discount_factor_gamma

    # get predicted Q for each action

    q_pred_all_actions = model(states)
    q_pred = batch_select_indices(q_pred_all_actions, actions)

    # get q_next

    q_next = ema_model(next_states)
    q_next = q_next.max(dim = -1).values

    # get target Q

    q_target_all_actions = ema_model(states)
    q_target = q_target_all_actions.max(dim = -1).values

    # main contribution of the paper is the following logic
    # section 4.1 - eq. 1

    # first take care of the loss for all actions except for the very last one

    q_pred_rest_actions, q_pred_last_action      = q_pred[..., :-1], q_pred[..., -1]
    q_target_first_action, q_target_rest_actions = q_target[..., 0], q_target[..., 1:]

    losses_all_actions_but_last = F.mse_loss(q_pred_rest_actions, q_target_rest_actions, reduction = 'none')

    # next take care of the very last action, which incorporates the rewards

    q_target_last_action, _ = pack([q_target_first_action[..., 1:], q_next], 'b *')

    q_target_last_action = rewards + γ * q_target_last_action

    losses_last_action = F.mse_loss(q_pred_last_action, q_target_last_action, reduction = 'none')

    # flatten and average

    losses, _ = pack([losses_all_actions_but_last, losses_last_action], '*')

    return losses.mean()

# main classes

class QRLHF(Module):
    @beartype
    def __init__(
        self,
        model: Module
    ):
        super().__init__()

    def forward(self):
        raise NotImplementedError
