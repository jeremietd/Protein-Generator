# Protein Generation using Tranception

Generation Steps:
1. Scores the sequence using Tranception
2. Generate possible mutations
3. Sample mutations
4. Mutate original sequence

Repeat until desired parameters

## Usage
```
python generator.py \
    --sequence [mdh_esm, mdh_esm_2, mdh_gan] \
    --model [small, medium, large] \
    --sampling_method [mirostat, top_k, top_p, typical] \
    --sampling_threshold (k for top-k, p for top-p, etc.) \
    --sequence_num (number of sequence to generate) \
    --evolution_cycles (number of mutation cycle) \
    --output_name folder_within_generated_sequence/sub_folder/fasta_file_name
```