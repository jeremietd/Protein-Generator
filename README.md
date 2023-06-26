# Protein Generation using Tranception

Generation Steps:
1. Scores the sequence using Tranception
2. Generate possible mutations
3. Sample mutations
4. Mutate original sequence
5. Repeat (1-4) until desired parameters

## Single-Mutant Usage
```
python generator.py \
    --sequence [mdh_esm, mdh_esm_2, avGFP] \
    --model [small, medium, large] \
    --sampling_method [mirostat, top_k, top_p, typical, random, greedy] \
    --sampling_threshold (k for top-k, p for top-p, etc.) \
    --sequence_num (number of sequence to generate) \
    --evolution_cycles (number of mutation cycle) \
    --save_df (optional, save metadata of generated sequences in dataframe) \
    --output_name folder_within_generated_sequence/sub_folder/fasta_file_name
```

## Multi-Mutant Usage
```
python generator.py \
    --sequence [mdh_esm, mdh_esm_2, avGFP] \
    --model [small, medium, large] \
    --mutations (number of multi-mutations) \
    --use_quantfun (use QFF for intermediate strategy) \
    --saved_model_dir [required if using QFF, path to saved model] \
    --intermediate_threshold (threshold for intermediate strategy) \
    --sampling_method [mirostat, top_k, top_p, typical, random, greedy] \
    --sampling_threshold (k for top-k, p for top-p, etc.) \
    --sequence_num (number of sequence to generate) \
    --evolution_cycles (number of mutation cycle) \
    --save_df (optional, save metadata of generated sequences in dataframe) \
    --output_name folder_within_generated_sequence/sub_folder/fasta_file_name
```

If using QFF, fine-tune the model first using the following command:
```
srun python Tranception-Protein-Generator/finetune_proteinbert.py \
    --train_data [path to training data] \
    --valid_data [path to validation/testing data] \
    --save_name [folder path for saving model] \
```
Train/Validation data needs to have 2 columns: 'seq' and 'label'. 'label' is the quantitative function value.