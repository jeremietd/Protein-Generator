import torch
import transformers
from transformers import PreTrainedTokenizerFast
import tranception
import datasets
from tranception import config, model_pytorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

AA_vocab = "ACDEFGHIKLMNPQRSTVWY"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "tranception/utils/tokenizers/Basic_tokenizer"),
                                                unk_token="[UNK]",
                                                sep_token="[SEP]",
                                                pad_token="[PAD]",
                                                cls_token="[CLS]",
                                                mask_token="[MASK]"
                                            )

def create_all_single_mutants(sequence,AA_vocab=AA_vocab,mutation_range_start=None,mutation_range_end=None):
  all_single_mutants={}
  sequence_list=list(sequence)
  if mutation_range_start is None: mutation_range_start=1
  if mutation_range_end is None: mutation_range_end=len(sequence)
  for position,current_AA in enumerate(sequence[mutation_range_start-1:mutation_range_end]):
    for mutated_AA in AA_vocab:
      if current_AA!=mutated_AA:
        mutated_sequence = sequence_list.copy()
        mutated_sequence[position] = mutated_AA
        all_single_mutants[current_AA+str(position+1)+mutated_AA]="".join(mutated_sequence)
  all_single_mutants = pd.DataFrame.from_dict(all_single_mutants,columns=['mutated_sequence'],orient='index')
  all_single_mutants.reset_index(inplace=True)
  all_single_mutants.columns = ['mutant','mutated_sequence']
  return all_single_mutants

def create_scoring_matrix_visual(scores,sequence,image_index=0,mutation_range_start=None,mutation_range_end=None,AA_vocab=AA_vocab,annotate=True,fontsize=20):
  filtered_scores=scores.copy()
  filtered_scores=filtered_scores[filtered_scores.position.isin(range(mutation_range_start,mutation_range_end+1))]
  piv=filtered_scores.pivot(index='position',columns='target_AA',values='avg_score').round(4)
  mutation_range_len = mutation_range_end - mutation_range_start + 1
  fig, ax = plt.subplots(figsize=(50,mutation_range_len))
  scores_dict = {}
  valid_mutant_set=set(filtered_scores.mutant)  
  ax.tick_params(bottom=True, top=True, left=True, right=True)
  ax.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True)
  if annotate:
    for position in range(mutation_range_start,mutation_range_end+1):
      for target_AA in list(AA_vocab):
        mutant = sequence[position-1]+str(position)+target_AA
        if mutant in valid_mutant_set:
          scores_dict[mutant]= float(filtered_scores.loc[filtered_scores.mutant==mutant,'avg_score'])
        else:
          scores_dict[mutant]=0.0
    labels = (np.asarray(["{} \n {:.4f}".format(symb,value) for symb, value in scores_dict.items() ])).reshape(mutation_range_len,len(AA_vocab))
    heat = sns.heatmap(piv,annot=labels,fmt="",cmap='RdYlGn',linewidths=0.30,ax=ax,vmin=np.percentile(scores.avg_score,2),vmax=np.percentile(scores.avg_score,98),\
                cbar_kws={'label': 'Log likelihood ratio (mutant / starting sequence)'},annot_kws={"size": fontsize})
  else:
    heat = sns.heatmap(piv,cmap='RdYlGn',linewidths=0.30,ax=ax,vmin=np.percentile(scores.avg_score,2),vmax=np.percentile(scores.avg_score,98),\
                cbar_kws={'label': 'Log likelihood ratio (mutant / starting sequence)'},annot_kws={"size": fontsize})
  heat.figure.axes[-1].yaxis.label.set_size(fontsize=int(fontsize*1.5))
  heat.figure.axes[-1].yaxis.set_ticklabels(heat.figure.axes[-1].yaxis.get_ticklabels(), fontsize=fontsize)
  heat.set_title("Higher predicted scores (green) imply higher protein fitness",fontsize=fontsize*2, pad=40)
  heat.set_ylabel("Sequence position", fontsize = fontsize*2)
  heat.set_xlabel("Amino Acid mutation", fontsize = fontsize*2)
  yticklabels = [str(pos)+' ('+sequence[pos-1]+')' for pos in range(mutation_range_start,mutation_range_end+1)]
  heat.set_yticklabels(yticklabels)
  heat.set_xticklabels(heat.get_xmajorticklabels(), fontsize = fontsize)
  heat.set_yticklabels(heat.get_ymajorticklabels(), fontsize = fontsize, rotation=0)
  plt.tight_layout()

  # Save output
  save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'scoring_matrix/')
  os.mkdir(save_path) if not os.path.exists(save_path) else None
  image_path = os.path.join(save_path, 'fitness_scoring_substitution_matrix_{}.png'.format(image_index))
  plt.savefig(image_path,dpi=100)
  # plt.show()

  return image_path

def suggest_mutations(scores):
  # intro_message = "The following mutations may be sensible options to improve fitness: \n\n"
  #Best mutants
  top_mutants=list(scores.sort_values(by=['avg_score'],ascending=False).head(5).mutant)
  top_mutants_fitness=list(scores.sort_values(by=['avg_score'],ascending=False).head(5).avg_score)
  top_mutants_recos = [top_mutant+" ("+str(round(top_mutant_fitness,4))+")" for (top_mutant,top_mutant_fitness) in zip(top_mutants,top_mutants_fitness)]
  # sorted_mutant_df = pd.DataFrame(list(zip(top_mutants, top_mutants_fitness)), columns =['top_mutants', 'top_mutants_score'])
  mutant_recos = "The single mutants with highest predicted fitness are (positive scores indicate fitness increase Vs starting sequence, negative scores indicate fitness decrease):\n {} \n\n".format(", ".join(top_mutants_recos))
  #Best positions
  positive_scores = scores[scores.avg_score > 0]
  positive_scores_position_avg = positive_scores.groupby(['position']).mean(numeric_only=True)
  top_positions=list(positive_scores_position_avg.sort_values(by=['avg_score'],ascending=False).head(5).index.astype(str))
  position_recos = "The positions with the highest average fitness increase are (only positions with at least one fitness increase are considered):\n {}".format(", ".join(top_positions))
  return print(intro_message+mutant_recos+position_recos)

def check_valid_mutant(sequence,mutant,AA_vocab=AA_vocab):
  valid = True
  try:
    from_AA, position, to_AA = mutant[0], int(mutant[1:-1]), mutant[-1]
  except:
    valid = False
  if sequence[position-1]!=from_AA: valid=False
  if position<1 or position>len(sequence): valid=False
  if to_AA not in AA_vocab: valid=False
  return valid

def get_mutated_protein(sequence,mutant):
  assert check_valid_mutant(sequence,mutant), "The mutant is not valid"
  mutated_sequence = list(sequence)
  mutated_sequence[int(mutant[1:-1])-1]=mutant[-1]
  return ''.join(mutated_sequence)

def score_and_create_matrix_all_singles(sequence,mutation_range_start=None,mutation_range_end=None,model_type="Small",scoring_mirror=False,batch_size_inference=20,max_number_positions_per_heatmap=50,num_workers=0,AA_vocab=AA_vocab, tokenizer=tokenizer, with_heatmap=True):
  if mutation_range_start is None: mutation_range_start=1
  if mutation_range_end is None: mutation_range_end=len(sequence)
  assert len(sequence) > 0, "no sequence entered"
  assert mutation_range_start <= mutation_range_end, "mutation range is invalid"
  if model_type=="Small":
    model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path="PascalNotin/Tranception_Small")
  elif model_type=="Medium":
    model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path="PascalNotin/Tranception_Medium")
  elif model_type=="Large":
    model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path="PascalNotin/Tranception_Large")
  if torch.cuda.is_available():
    model.cuda()
    print("Inference will take place on GPU")
  else:
    print("Inference will take place on CPU")
  model.config.tokenizer = tokenizer
  all_single_mutants = create_all_single_mutants(sequence,AA_vocab,mutation_range_start,mutation_range_end)
  scores = model.score_mutants(DMS_data=all_single_mutants, 
                                    target_seq=sequence, 
                                    scoring_mirror=scoring_mirror, 
                                    batch_size_inference=batch_size_inference,  
                                    num_workers=num_workers, 
                                    indel_mode=False
                                    )
  scores = pd.merge(scores,all_single_mutants,on="mutated_sequence",how="left")
  scores["position"]=scores["mutant"].map(lambda x: int(x[1:-1]))
  scores["target_AA"] = scores["mutant"].map(lambda x: x[-1])
  score_heatmaps = []
  if with_heatmap:
    mutation_range = mutation_range_end - mutation_range_start + 1
    number_heatmaps = int((mutation_range - 1) / max_number_positions_per_heatmap) + 1
    image_index = 0
    window_start = mutation_range_start
    window_end = min(mutation_range_end,mutation_range_start+max_number_positions_per_heatmap-1)
    for image_index in range(number_heatmaps):
      score_heatmaps.append(create_scoring_matrix_visual(scores,sequence,image_index,window_start,window_end,AA_vocab))
      window_start += max_number_positions_per_heatmap
      window_end = min(mutation_range_end,window_start+max_number_positions_per_heatmap-1)
  return score_heatmaps, suggest_mutations(scores), scores

def extract_sequence(example):
  label, taxon, sequence = example
  return sequence

def clear_inputs(protein_sequence_input,mutation_range_start,mutation_range_end):
  protein_sequence_input = ""
  mutation_range_start = None
  mutation_range_end = None
  return protein_sequence_input,mutation_range_start,mutation_range_end
