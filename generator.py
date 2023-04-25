import app
import argparse
from transformers import PreTrainedTokenizerFast
import pandas as pd
import os
from sampling import top_k_sampling, temperature_sampler, top_p_sampling, typical_sampling, mirostat_sampling

parser = argparse.ArgumentParser()
parser.add_argument('--sequence', type=str, choices=["brca1", "adrb2", "p53"], default='brca1')
parser.add_argument('--mutation_start', type=int, default=None)
parser.add_argument('--mutation_end', type=int, default=None)
parser.add_argument('--model', type=str, choices=['small', 'medium', 'large'], default='small')
parser.add_argument('--use_scoring_mirror', action='store_true', help='Whether to score the sequence from both ends')
parser.add_argument('--batch', type=int, default=20, help='Batch size for scoring')
parser.add_argument('--max_pos', type=int, default=50, help='Maximum number of positions per heatmap')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
parser.add_argument('--with_heatmap', action='store_true', help='Whether to generate heatmap')
parser.add_argument('--save_scores', action='store_true', help='Whether to save scores')
args = parser.parse_args()

AA_vocab = "ACDEFGHIKLMNPQRSTVWY"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "tranception/utils/tokenizers/Basic_tokenizer"),
                                                unk_token="[UNK]",
                                                sep_token="[SEP]",
                                                pad_token="[PAD]",
                                                cls_token="[CLS]",
                                                mask_token="[MASK]"
                                            )

example_sequence = {'BRCA1_HUMAN': 'MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLKLLNQKKGPSQCPLCKNDITKRSLQESTRFSQLVEELLKIICAFQLDTGLEYANSYNFAKKENNSPEHLKDEVSIIQSMGYRNRAKRLLQSEPENPSLQETSLSVQLSNLGTVRTLRTKQRIQPQKTSVYIELGSDSSEDTVNKATYCSVGDQELLQITPQGTRDEISLDSAKKAACEFSETDVTNTEHHQPSNNDLNTTEKRAAERHPEKYQGSSVSNLHVEPCGTNTHASSLQHENSSLLLTKDRMNVEKAEFCNKSKQPGLARSQHNRWAGSKETCNDRRTPSTEKKVDLNADPLCERKEWNKQKLPCSENPRDTEDVPWITLNSSIQKVNEWFSRSDELLGSDDSHDGESESNAKVADVLDVLNEVDEYSGSSEKIDLLASDPHEALICKSERVHSKSVESNIEDKIFGKTYRKKASLPNLSHVTENLIIGAFVTEPQIIQERPLTNKLKRKRRPTSGLHPEDFIKKADLAVQKTPEMINQGTNQTEQNGQVMNITNSGHENKTKGDSIQNEKNPNPIESLEKESAFKTKAEPISSSISNMELELNIHNSKAPKKNRLRRKSSTRHIHALELVVSRNLSPPNCTELQIDSCSSSEEIKKKKYNQMPVRHSRNLQLMEGKEPATGAKKSNKPNEQTSKRHDSDTFPELKLTNAPGSFTKCSNTSELKEFVNPSLPREEKEEKLETVKVSNNAEDPKDLMLSGERVLQTERSVESSSISLVPGTDYGTQESISLLEVSTLGKAKTEPNKCVSQCAAFENPKGLIHGCSKDNRNDTEGFKYPLGHEVNHSRETSIEMEESELDAQYLQNTFKVSKRQSFAPFSNPGNAEEECATFSAHSGSLKKQSPKVTFECEQKEENQGKNESNIKPVQTVNITAGFPVVGQKDKPVDNAKCSIKGGSRFCLSSQFRGNETGLITPNKHGLLQNPYRIPPLFPIKSFVKTKCKKNLLEENFEEHSMSPEREMGNENIPSTVSTISRNNIRENVFKEASSSNINEVGSSTNEVGSSINEIGSSDENIQAELGRNRGPKLNAMLRLGVLQPEVYKQSLPGSNCKHPEIKKQEYEEVVQTVNTDFSPYLISDNLEQPMGSSHASQVCSETPDDLLDDGEIKEDTSFAENDIKESSAVFSKSVQKGELSRSPSPFTHTHLAQGYRRGAKKLESSEENLSSEDEELPCFQHLLFGKVNNIPSQSTRHSTVATECLSKNTEENLLSLKNSLNDCSNQVILAKASQEHHLSEETKCSASLFSSQCSELEDLTANTNTQDPFLIGSSKQMRHQSESQGVGLSDKELVSDDEERGTGLEENNQEEQSMDSNLGEAASGCESETSVSEDCSGLSSQSDILTTQQRDTMQHNLIKLQQEMAELEAVLEQHGSQPSNSYPSIISDSSALEDLRNPEQSTSEKAVLTSQKSSEYPISQNPEGLSADKFEVSADSSTSKNKEPGVERSSPSKCPSLDDRWYMHSCSGSLQNRNYPSQEELIKVVDVEEQQLEESGPHDLTETSYLPRQDLEGTPYLESGISLFSDDPESDPSEDRAPESARVGNIPSSTSALKVPQLKVAESAQSPAAAHTTDTAGYNAMEESVSREKPELTASTERVNKRMSMVVSGLTPEEFMLVYKFARKHHITLTNLITEETTHVVMKTDAEFVCERTLKYFLGIAGGKWVVSYFWVTQSIKERKMLNEHDFEVRGDVVNGRNHQGPKRARESQDRKIFRGLEICCYGPFTNMPTDQLEWMVQLCGASVVKELSSFTLGTGVHPIVVVQPDAWTEDNGFHAIGQMCEAPVVTREWVLDSVALYQCQELDTYLIPQIPHSHY',
                    'ADRB2_HUMAN':'MGQPGNGSAFLLAPNGSHAPDHDVTQERDEVWVVGMGIVMSLIVLAIVFGNVLVITAIAKFERLQTVTNYFITSLACADLVMGLAVVPFGAAHILMKMWTFGNFWCEFWTSIDVLCVTASIETLCVIAVDRYFAITSPFKYQSLLTKNKARVIILMVWIVSGLTSFLPIQMHWYRATHQEAINCYANETCCDFFTNQAYAIASSIVSFYVPLVIMVFVYSRVFQEAKRQLQKIDKSEGRFHVQNLSQVEQDGRTGHGLRRSSKFCLKEHKALKTLGIIMGTFTLCWLPFFIVNIVHVIQDNLIRKEVYILLNWIGYVNSGFNPLIYCRSPDFRIAFQELLCLRRSSLKAYGNGYSSNGNTGEQSGYHVEQEKENKLLCEDLPGTEDFVGHQGTVPSDNIDSQGRNCSTNDSLL',
                    'P53_HUMAN': 'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPRVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD'}

seq = example_sequence.get(args.sequence.upper() + "_HUMAN")

mutation_start = args.mutation_start
mutation_end = args.mutation_end
model = args.model.capitalize()

# 1. Get scores of suggested mutation
score_heatmap, suggested_mutation, scores = app.score_and_create_matrix_all_singles(seq, mutation_start, mutation_end, 
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
    pd.DataFrame(scores, columns =['score_scores']).to_csv(save_path_scores)
    print(f"Results saved to {save_path_scores}")

# 2. Sample mutation from suggested mutation scores
temp_sampler = temperature_sampler(1.0)

topk_mutation = top_k_sampling(scores, k=5, sampler=temp_sampler)
print("Top-K sampled mutation: ", topk_mutation)

topp_mutation = top_p_sampling(scores, p=0.9, sampler=temp_sampler)
print("Top-P sampled mutation: ", topp_mutation)

typical_mutation = typical_sampling(scores, mass=0.95, sampler=temp_sampler)
print("Typically sampled mutation: ", typical_mutation)

mirostat_mutation = mirostat_sampling(scores, tau=3.0, sampler=temp_sampler)
print("Typically sampled mutation: ", mirostat_mutation)

# 3. Get Mutated Sequence
# mutated_sequence = app.get_mutated_sequence(seq, mutation)