import random
import math
import app
import pandas as pd

AA_vocab = "ACDEFGHIKLMNPQRSTVWY"

class UCTNode:
    def __init__(self, state, parent=None):
        self.state = state # state is a sequence
        self.parent = parent # parent is a node
        self.children = [] # children is a list of nodes
        self.is_expanded = False
        self.visits = 0
        self.rewards = 0

def select(node): # TODO: Check input type requirements
    """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
        lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
        exploration versus exploitation.
        Source: https://github.com/A-Malone/Monte-carlo-tree-search/blob/master/UCT.py
    """
    s = node
    while node.is_expanded:
        print("Children Node:", node.children)
        s = sorted(node.children, key = lambda c: c.rewards/c.visits + sqrt(2*log(node.visits)/c.visits))[-1]
    return s

def expand(node, AA_vocab):
    # Implement the expansion step by adding child nodes for unexplored actions
    node.is_expanded = True
    extension = app.extend_sequence_by_n(node.state, 1, reference_vocab=AA_vocab, output_sequence=False)
    next = extension.sample(1) #TODO: Random or topk?
    next_str = next['extension'].to_string(index=False)
    return node, UCTNode(str(next_str)) # child of root_node

def simulate(root_node, node, model, tokenizer, AA_vocab):
    # Perform random simulations (rollouts) from the selected node until a terminal state is reached
    # NextNode = expand(node, AA_vocab)
    # print("node: ", node.state)
    # print("Next node: ", NextNode.state)
    root_node.children.append(node)
    # print("Children: ", node.children)
    seq = [root_node.state + node.state]
    print("Sequence: ", seq)
        
    df_es = pd.DataFrame.from_dict({"mutated_sequence": seq})
    results, _ = app.score_multi_mutations(sequence=None, extra_mutants=df_es, model_type=model, scoring_mirror=False, batch_size_inference=1, max_number_positions_per_heatmap=50, num_workers=8, AA_vocab=AA_vocab, tokenizer=tokenizer, AR_mode=True)
    return results['avg_score']

def backpropagate(root_node, node, reward):
    # Update the node statistics (visits and rewards) along the path from the selected node to the root
    while node is not None:
        node.visits += 1
        node.rewards += reward
        node.parent = root_node

def mcts(root_state: str, max_length, model_type, tokenizer, AA_vocab):
    root_node = UCTNode(root_state)
    # Maybe expand the root node here?

    for _ in range(max_length):
        print("Root node: ", root_node.state)
        selected_node = select(root_node)

        print("Selected node: ", selected_node.state)
        root_node, expanded_node = expand(selected_node, AA_vocab)

        print("Expanded node: ", expanded_node.state)
        reward = simulate(root_node, expanded_node, model_type, tokenizer, AA_vocab)

        backpropagate(root_node, expanded_node, reward)
        # root_node = expanded_node

    # Return the best child node's state as the chosen action
    best_child = max(root_node.children, key=lambda n: n.visits)
    return root_state + best_child.state
