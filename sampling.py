import torch
from torch.distributions import Categorical
import pandas as pd

class temperature_sampler:
  def __init__(self, temperature: float = 1.0):
    self.temperature = temperature
  def __call__(self, logits: torch.Tensor):
    dist = Categorical(logits=logits / self.temperature)
    return dist.sample()

def top_k_sampling(k, scores: pd.DataFrame, sampler = temperature_sampler(temperature=1.0)):
  raw_score = torch.tensor(scores['avg_score'].values)
  zeros = raw_score.new_ones(raw_score.shape) * float('-inf')
  values, indices = torch.topk(raw_score, k=k, dim=-1)
  zeros.scatter_(-1, indices, values)
  
  sampled_score = sampler(zeros)
  index = (zeros == sampled_score).nonzero(as_tuple=True)[0]

  return scores['mutant'][index]

def typical_sampling(scores, filter_value: float = float("-inf"), mass: float = 0.9, min_tokens_to_keep: int = 1):
  # calculate entropy
  normalized = torch.nn.functional.log_softmax(scores, dim=-1)
  p = torch.exp(normalized)
  ent = -(normalized * p).nansum(-1, keepdim=True)

  # shift and sort
  shifted_scores = torch.abs((-normalized) - ent)
  sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
  sorted_logits = scores.gather(-1, sorted_indices)
  cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

  # Remove tokens with cumulative mass above the threshold
  last_ind = (cumulative_probs < mass).sum(dim=1)
  last_ind[last_ind < 0] = 0
  sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
  if min_tokens_to_keep > 1:
      # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
      sorted_indices_to_remove[..., : min_tokens_to_keep] = 0
  indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

  scores = scores.masked_fill(indices_to_remove, filter_value)
  return scores