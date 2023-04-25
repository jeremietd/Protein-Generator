import torch
from torch.distributions import Categorical
import pandas as pd

class temperature_sampler:
  def __init__(self, temperature: float = 1.0):
    self.temperature = temperature
  def __call__(self, logits: torch.Tensor):
    dist = Categorical(logits=logits / self.temperature)
    return dist.sample()

def top_k_sampling(scores: pd.DataFrame, k: int, sampler = temperature_sampler(temperature=1.0)):
  raw_score = torch.tensor(scores['avg_score'].values)
  raw_score = torch.nan_to_num(raw_score, float("-inf"))
  zeros = raw_score.new_ones(raw_score.shape) * float('-inf')
  values, indices = torch.topk(raw_score, k=k, dim=-1)
  zeros.scatter_(-1, indices, values)
  
  sampled_score = sampler(zeros).item()

  return scores['mutant'][sampled_score]

def typical_sampling(scores: pd.DataFrame, mass: float = 0.9, sampler = temperature_sampler(temperature=1.0)):
  raw_score = torch.tensor(scores['avg_score'].values)
  raw_score = torch.nan_to_num(raw_score, float("-inf"))
  # calculate entropy
  normalized = torch.nn.functional.log_softmax(raw_score, dim=-1)
  p = torch.exp(normalized)
  ent = -(normalized * p).nansum(-1, keepdim=True)

  # shift and sort
  shifted_scores = torch.abs((-normalized) - ent)
  sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
  sorted_logits = raw_score.gather(-1, sorted_indices)
  cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

  # Remove tokens with cumulative mass above the threshold
  last_ind = (cumulative_probs < mass).sum(dim=-1)
  last_ind[last_ind < 0] = 0
  sorted_indices_to_remove = sorted_scores > sorted_scores.gather(-1, last_ind.view(-1))
  indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)

  raw_score = raw_score.masked_fill(indices_to_remove, float("-inf"))
  sampled_score = sampler(raw_score).item()
  return scores['mutant'][sampled_score]

def top_p_sampling(scores: pd.DataFrame, p: float, sampler = temperature_sampler(temperature=1.0)):
  raw_score = torch.tensor(scores['avg_score'].values)
  raw_score = torch.nan_to_num(raw_score, float("-inf"))
  
  sorted_logits, sorted_indices = torch.sort(raw_score, dim=-1, descending=True)
  cumulative_probs = torch.cumsum(sorted_logits, dim=-1)

  nucleus = cumulative_probs > p
 # Shift the indices to the right to keep also the first token above the threshold
  nucleus[..., 1:] = nucleus[..., :-1].clone()
  nucleus[..., 0] = 0
  indices_to_remove = nucleus.scatter(-1, sorted_indices, nucleus)
  raw_score = raw_score.masked_fill(indices_to_remove, float("-inf"))
  sampled_score = sampler(raw_score).item()

  # return res
  return scores['mutant'][sampled_score]