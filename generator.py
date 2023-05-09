import app
import argparse
from transformers import PreTrainedTokenizerFast
import pandas as pd
import os
import util
from sampling import top_k_sampling, temperature_sampler, top_p_sampling, typical_sampling, mirostat_sampling

parser = argparse.ArgumentParser()
parser.add_argument('--sequence', type=str, choices=["mdh_esm", "mdh_esm_2", "mdh_gan"], default='mdh_esm', help='Sequence to do mutation or DE')
parser.add_argument('--mutation_start', type=int, default=None, help='Mutation start position')
parser.add_argument('--mutation_end', type=int, default=None, help='Mutation end position')
parser.add_argument('--model', type=str, choices=['small', 'medium', 'large'], default='small', help='Tranception model size')
parser.add_argument('--use_scoring_mirror', action='store_true', help='Whether to score the sequence from both ends')
parser.add_argument('--batch', type=int, default=20, help='Batch size for scoring')
parser.add_argument('--max_pos', type=int, default=50, help='Maximum number of positions per heatmap')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
parser.add_argument('--with_heatmap', action='store_true', help='Whether to generate heatmap')
parser.add_argument('--save_scores', action='store_true', help='Whether to save scores')

parser.add_argument('--sampling_method', type=str, choices=['top_k', 'top_p', 'typical', 'mirostat'], required=True, help='Sampling method')
parser.add_argument('--sampling_threshold', type=float, required=True, help='Sampling threshold (k for top_k, p for top_p, tau for mirostat, etc.)')
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
                    'MDH_A0A522AU97': 'MNRPRVTVVGAGNVGATTAQRIAEAGLADVVLVDIVEGLPQGKALDLAEAAPVMGHDARITGTNDYADTAGSSIVVVTSGLARQPGMSRDDLLAKNAGIVRSVVEQAVKHSPDCILIIVTNPLDAMCHVAMEASGFPRERVIGMAGVLDSARFRTFIAMELGVSVEDTHAFVLGGHGDTMVPLPRYSTVAGIPITELMTPERVAALVDRTANGGAEIVALLKAGSAYYAPAASTYEMVDAILGDRKRVLPCAVHLEGEYGVNGLFVGVPVVLGARGMERIIEIDLTAEENTAFQRSAAAVQELVDKLHV'}

mutation_start = args.mutation_start
mutation_end = args.mutation_end
model = args.model.capitalize()
sequence_num = args.sequence_num
evolution_cycles = args.evolution_cycles
generated_sequence = []
sequence_iteration = []
generated_sequence_name = []

while len(generated_sequence) < sequence_num:

    iteration = 0
    if args.sequence == 'mdh_esm':
        seq = example_sequence.get('MDH_A0A075B5H0')
        sequence_id = 'MDH_A0A075B5H0'
    elif args.sequence == 'mdh_esm_2':
        seq = example_sequence.get('MDH_A0A2V9QQ45')
        sequence_id = 'MDH_A0A2V9QQ45'
    elif args.sequence == 'mdh_gan':
        seq = example_sequence.get('MDH_A0A522AU97')
        sequence_id = 'MDH_A0A522AU97'

    while iteration < evolution_cycles:
        print(f"Sequence {len(generated_sequence) + 1} of {sequence_num}, Iteration {iteration + 1} of {evolution_cycles}")
        # 1. Get scores of suggested mutation
        score_heatmap, suggested_mutation, scores, _ = app.score_and_create_matrix_all_singles(seq, mutation_start, mutation_end, 
                                                                                    model, 
                                                                                    scoring_mirror=args.use_scoring_mirror, 
                                                                                    batch_size_inference=args.batch, 
                                                                                    max_number_positions_per_heatmap=args.max_pos, 
                                                                                    num_workers=args.num_workers, 
                                                                                    AA_vocab=AA_vocab, 
                                                                                    tokenizer=tokenizer,
                                                                                    with_heatmap=args.with_heatmap)

        # Save heatmap
        if args.with_heatmap and args.save_scores:
            save_path_heatmap = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output_heatmap.csv")
            pd.DataFrame(score_heatmap, columns =['score_heatmap']).to_csv(save_path_heatmap)
            print(f"Results saved to {save_path_heatmap}")

        # Save scores
        if args.save_scores:
            save_path_scores = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output_scores.csv")
            scores.to_csv(save_path_scores)
            print(f"Scores saved to {save_path_scores}")

        # 2. Sample mutation from suggested mutation scores
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
        print(f"Using {sampling_strat} sampling strategy with threshold {sampling_threshold}")

        # 3. Get Mutated Sequence
        mutated_sequence = app.get_mutated_protein(seq, mutation)

        print("Original Sequence: ", seq)
        print("Mutation: ", mutation)
        print("Mutated Sequence: ", mutated_sequence)

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