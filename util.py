import app

def add_mutation(mutate_dict, seq, seq_name, suggested_mutation):
    mutated_seq = app.get_mutated_protein(seq, suggested_mutation)
    if suggested_mutation not in mutate_dict:
        mutate_dict[suggested_mutation] = dict()
    mutate_dict[suggested_mutation][seq_name] = mutated_seq