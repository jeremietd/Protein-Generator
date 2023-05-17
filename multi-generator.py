import app
import argparse
from transformers import PreTrainedTokenizerFast
import pandas as pd
import os
import util
import tensorflow as tf
from sampling import top_k_sampling, temperature_sampler, top_p_sampling, typical_sampling, mirostat_sampling
from proteinbert.model_generation import InputEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--sequence', type=str, choices=["mdh_esm", "mdh_esm_2", "avGFP"], default='mdh_esm', help='Sequence to do mutation or DE')
parser.add_argument('--mutation_start', type=int, default=None, help='Mutation start position')
parser.add_argument('--mutation_end', type=int, default=None, help='Mutation end position')
parser.add_argument('--model', type=str, choices=['small', 'medium', 'large'], default='small', help='Tranception model size')
parser.add_argument('--use_scoring_mirror', action='store_true', help='Whether to score the sequence from both ends')
parser.add_argument('--batch', type=int, default=20, help='Batch size for scoring')
parser.add_argument('--max_pos', type=int, default=50, help='Maximum number of positions per heatmap')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')
parser.add_argument('--with_heatmap', action='store_true', help='Whether to generate heatmap')
parser.add_argument('--mutations', type=int, default=2, help='Number of mutations to generate')
parser.add_argument('--save_scores', action='store_true', help='Whether to save scores')

parser.add_argument('--sampling_method', type=str, choices=['top_k', 'top_p', 'typical', 'mirostat'], required=True, help='Sampling method')
parser.add_argument('--sampling_threshold', type=float, required=True, help='Sampling threshold (k for top_k, p for top_p, tau for mirostat, etc.)')
parser.add_argument('--intermediate_threshold', type=int, help='Top-K threshold for intermediate sampling')
parser.add_argument('--use_proteinbert', action='store_true', help='Whether to use ProteinBERT for sampling')
parser.add_argument('--saved_model_dir', type=str, help='ProteinBERT saved model directory')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for final sampling; 1.0 equals to random sampling')
parser.add_argument('--sequence_num', type=int, required=True, help='Number of sequences to generate')
parser.add_argument('--evolution_cycles', type=int, required=True, help='Number of evolution cycles per generated sequence')
parser.add_argument('--output_name', type=str, required=True, help='Output file name (Just name with no extension!)')
args = parser.parse_args()

AA_vocab = "ACDEFGHIKLMNPQRSTVWY"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "tranception/utils/tokenizers/Basic_tokenizer"),
                                                unk_token="[UNK]",
                                                sep_token="[SEP]",
                                                pad_token="[PAD]",
                                                cls_token="[CLS]",
                                                mask_token="[MASK]"
                                            )

example_sequence = {'MDH_A0A075B5H0': 'MTQRKKISLIGAGNIGGTLAHLIAQKELGDVVLFDIVEGMPQGKALDISHSSPIMGSNVKITGTNNYEDIKGSDVVIITAGIPRKPGKSDKEWSRDDLLSVNAKIMKDVAENIKKYCPNAFVIVVTNPLDVMVYVLHKYSGLPHNKVCGMAGVLDSSRFRYFLAEKLNVSPNDVQAMVIGGHGDTMVPLTRYCTVGGIPLTEFIKQGWITQEEIDEIVERTRNAGGEIVNLLKTGSAYFAPAASAIEMAESYLKDKKRILPCSAYLEGQYGVKDLFVGVPVIIGKNGVEKIIELELTEEEQEMFDKSVESVRELVETVKKLNALEHHHHHH',
                    'MDH_A0A2V9QQ45': 'MRKKVTIVGSGNVGATAAQRIVDKELADVVLIDIIEGVPQGKGLDLLQSGPIEGYDSHVLGTNDYKDTANSDIVVITAGLPRRPGMSRDDLLIKNYEIVKGVTEQVVKYSPHSILIVVSNPLDAMVQTAFKISGFPKNRVIGMAGVLDSARFRTFIAMELNVSVENIHAFVLGGHGDTMVPLPRYSTVAGIPITELLPRERIDALVKRTRDGGAEIVGLLKTGSAYYAPSAATVEMVEAIFKDKKKILPCAAYLEGEYGISGSYVGVPVKLGKSGVEEIIQIKLTPEENAALKKSANAVKELVDIIKV',
                    'avGFP': 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'}

mutation_start = args.mutation_start
mutation_end = args.mutation_end
model = args.model.capitalize()
sequence_num = args.sequence_num
evolution_cycles = args.evolution_cycles
generated_sequence = []
sequence_iteration = []
generated_sequence_name = []

assert args.intermediate_sampling_threshold <= 100, "Intermediate sampling threshold cannot be greater than 100!"
if args.use_proteinbert:
    assert args.saved_model_dir is not None, "Please specify the saved model directory for ProteinBERT!"
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"{args.saved_model_dir}")
    
    input_encoder = InputEncoder(n_annotations=8943) # Check this number
    proteinbert_model = app.load_savedmodel(model_path=model_path)
    print("ProteinBERT model will be used!")
else:
    assert args.intermediate_threshold is not None, "Please specify the intermediate threshold for Top-K sampling!"
    print("Intermediate Top-K sampling will be used!")

while len(generated_sequence) < sequence_num:

    iteration = 0
    if args.sequence == 'mdh_esm':
        seq = example_sequence.get('MDH_A0A075B5H0')
        sequence_id = 'MDH_A0A075B5H0'
    elif args.sequence == 'mdh_esm_2':
        seq = example_sequence.get('MDH_A0A2V9QQ45')
        sequence_id = 'MDH_A0A2V9QQ45'
    elif args.sequence == 'avGFP':
        seq = example_sequence.get('avGFP')
        sequence_id = 'avGFP'

    while iteration < evolution_cycles:
        print(f"Sequence {len(generated_sequence) + 1} of {sequence_num}, Iteration {iteration + 1} of {evolution_cycles}")
        print("=========================================")

        mutation_count = 0
        while mutation_count < args.mutations:
            mutation_count += 1
            print(f"Mutation {mutation_count} of {args.mutations}")
            # First Mutation
            if mutation_count == 1:
                # 1. Get scores of suggested mutation
                score_heatmap, suggested_mutation, scores, single_DMS = app.score_and_create_matrix_all_singles(seq, mutation_start, mutation_end, 
                                                                                            model, 
                                                                                            scoring_mirror=args.use_scoring_mirror, 
                                                                                            batch_size_inference=args.batch, 
                                                                                            max_number_positions_per_heatmap=args.max_pos, 
                                                                                            num_workers=args.num_workers, 
                                                                                            AA_vocab=AA_vocab, 
                                                                                            tokenizer=tokenizer,
                                                                                            with_heatmap=args.with_heatmap)

                last_round_DMS = single_DMS
                # Save heatmap
                if args.with_heatmap and args.save_scores:
                    save_path_heatmap = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"output_heatmap_round_{mutation_count}.csv")
                    pd.DataFrame(score_heatmap, columns =['score_heatmap']).to_csv(save_path_heatmap)
                    print(f"Results saved to {save_path_heatmap}")

                # Save scores
                if args.save_scores:
                    save_path_scores = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output_scores.csv")
                    scores.to_csv(save_path_scores)
                    print(f"Scores saved to {save_path_scores}")

                # 2. Sample mutation from suggested mutation scores
                final_sampler = temperature_sampler(args.temperature)
                intermediate_sampling_threshold = args.intermediate_threshold
                assert intermediate_sampling_threshold > 0, "Intermediate sampling threshold must be greater than 0!"
            
            # Subsequent Mutations
            if mutation_count > 1 and mutation_count < args.mutations:

                last_mutation_round_DMS = last_round_DMS
                print(f"Generating 1 extra mutations after {len(last_mutation_round_DMS['mutant'][0].split(':'))} rounds to make {mutation_count} rounds in total")
                assert len(last_mutation_round_DMS['mutant'][0].split(':')) == mutation_count-1, "Mutation step not consistent with previous mutation round"
                all_extra_mutants = app.generate_n_extra_mutations(DMS_data=last_mutation_round_DMS, extra_mutations=1)
                
                if args.use_proteinbert:
                    all_extra_mutants = all_extra_mutants.sample(n=100)
                    extra_mutants = app.predict_proteinBERT(model=proteinbert_model, DMS=all_extra_mutants,input_encoder=input_encoder, top_n=intermediate_sampling_threshold, batch_size=128)
                else:
                    mutation = top_k_sampling(scores, k=int(100), sampler=final_sampler, multi=True)
                    trimmed = app.trim_DMS(DMS_data=all_extra_mutants, sampled_mutants=mutation, mutation_rounds=mutation_count)
                    extra_mutants = trimmed.sample(n=intermediate_sampling_threshold)
                print(f"Using {len(extra_mutants)} variants for scoring")

                # 1. Get scores of suggested mutation
                suggested_mutation, scores, extra_DMS = app.score_multi_mutations(seq,
                                                                                extra_mutants=extra_mutants,
                                                                                mutation_range_start=mutation_start, 
                                                                                mutation_range_end=mutation_end, 
                                                                                model_type=model, 
                                                                                scoring_mirror=args.use_scoring_mirror, 
                                                                                batch_size_inference=args.batch, 
                                                                                max_number_positions_per_heatmap=args.max_pos, 
                                                                                num_workers=args.num_workers, 
                                                                                AA_vocab=AA_vocab, 
                                                                                tokenizer=tokenizer)

                last_round_DMS = extra_DMS
                # Save scores
                if args.save_scores:
                    save_path_scores = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"output_scores_round_{extra_mutation_rounds}.csv")
                    scores.to_csv(save_path_scores)
                    print(f"Scores saved to {save_path_scores}")

                # 2. Intermediate sample mutation from suggested mutation scores
                final_sampler = temperature_sampler(args.temperature)
                intermediate_sampling_threshold = args.intermediate_threshold
                assert intermediate_sampling_threshold > 0, "Intermediate sampling threshold must be greater than 0!"

                # if not args.use_proteinbert:
                #     print(f"Using Top-{intermediate_sampling_threshold} as intermediate sampling strategy")
                #     mutation = top_k_sampling(scores, k=int(intermediate_sampling_threshold), sampler=final_sampler, multi=True)

            # Last Mutation
            if mutation_count == args.mutations:
                # 1. Get scores of suggested mutation
                last_mutation_round_DMS = last_round_DMS
                print(f"Generating 1 extra mutations after {len(last_mutation_round_DMS['mutant'][0].split(':'))} rounds to make {mutation_count} rounds in total")
                assert len(last_mutation_round_DMS['mutant'][0].split(':')) == mutation_count-1, "Mutation step not consistent with previous mutation round"
                all_extra_mutants = app.generate_n_extra_mutations(DMS_data=last_mutation_round_DMS, extra_mutations=1)

                if args.use_proteinbert:
                    all_extra_mutants = all_extra_mutants.sample(n=100)
                    extra_mutants = app.predict_proteinBERT(model=proteinbert_model, DMS=all_extra_mutants,input_encoder=input_encoder, top_n=intermediate_sampling_threshold, batch_size=128)
                else:
                    mutation = top_k_sampling(scores, k=int(100), sampler=final_sampler, multi=True)
                    trimmed = app.trim_DMS(DMS_data=all_extra_mutants, sampled_mutants=mutation, mutation_rounds=mutation_count)
                    extra_mutants = trimmed.sample(n=intermediate_sampling_threshold)
                print(f"Using {len(extra_mutants)} variants for scoring")

                # 1. Get scores of suggested mutation
                suggested_mutation, scores, extra_DMS = app.score_multi_mutations(seq,
                                                                                extra_mutants=extra_mutants,
                                                                                mutation_range_start=mutation_start, 
                                                                                mutation_range_end=mutation_end, 
                                                                                model_type=model, 
                                                                                scoring_mirror=args.use_scoring_mirror, 
                                                                                batch_size_inference=args.batch, 
                                                                                max_number_positions_per_heatmap=args.max_pos, 
                                                                                num_workers=args.num_workers, 
                                                                                AA_vocab=AA_vocab, 
                                                                                tokenizer=tokenizer)

                # Save scores
                if args.save_scores:
                    save_path_scores = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"output_scores_round_{extra_mutation_rounds}.csv")
                    scores.to_csv(save_path_scores)
                    print(f"Scores saved to {save_path_scores}")

                # 2. Final Sampling mutation from suggested mutation scores
                final_sampler = temperature_sampler(args.temperature)
                sampling_strat = args.sampling_method
                sampling_threshold = args.sampling_threshold

                if sampling_strat == 'top_k':
                    mutation = top_k_sampling(scores, k=int(sampling_threshold), sampler=final_sampler)
                elif sampling_strat == 'top_p':
                    assert float(sampling_threshold) <= 1.0 and float(sampling_threshold) > 0, "Top-p sampling threshold must be between 0 and 1"
                    mutation = top_p_sampling(scores, p=float(sampling_threshold), sampler=final_sampler)
                elif sampling_strat == 'typical':
                    assert float(sampling_threshold) < 1.0 and float(sampling_threshold) > 0, "Typical sampling threshold must be between 0 and 1"
                    mutation = typical_sampling(scores, mass=float(sampling_threshold), sampler=final_sampler)
                elif sampling_strat == 'mirostat':
                    mutation = mirostat_sampling(scores, tau=float(sampling_threshold), sampler=final_sampler)
                else:
                    raise ValueError(f"Sampling strategy {sampling_strat} not supported")
                print(f"Using {sampling_strat} as final sampling strategy with threshold {sampling_threshold}")

        # 3. Get Mutated Sequence
        mutated_sequence = app.get_mutated_protein(seq, mutation)

        print("Original Sequence: ", seq)
        print("Mutation: ", mutation)
        print("Mutated Sequence: ", mutated_sequence)
        print("=========================================")

        seq = mutated_sequence

        iteration += 1

    generated_sequence.append(mutated_sequence)
    sequence_iteration.append(iteration)
    seq_name = 'Tranception_{}_{}x_{}'.format(sequence_id, iteration, len(generated_sequence))
    generated_sequence_name.append(seq_name)
    

generated_sequence_df = pd.DataFrame({'name': generated_sequence_name,'sequence': generated_sequence, 'iterations': sequence_iteration})
save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "generated_sequence/{}.fasta".format(args.output_name))
os.makedirs(os.path.dirname(os.path.realpath(save_path))) if not os.path.exists(os.path.dirname(os.path.realpath(save_path))) else None

util.save_as_fasta(generated_sequence_df, save_path)
print(f"Generated sequences saved to {save_path}")