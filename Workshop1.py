import random
import math 

def create_sequence():
    nucleotid_bases = ["A", "C", "G", "T"]
    size_sequence = random.randint(10, 20)
    new_sequence = [nucleotid_bases[random.randint(0, 3)] for i in range(size_sequence)]
    return "".join(new_sequence)

def create_database():
    db_size = 10
    data_base = [create_sequence() for i in range(db_size)]
    return data_base

def create_dataset(dataset_size: int):
    dataset = [create_sequence() for i in range(dataset_size)]
    return dataset

def get_combinations(n, sequences, bases):
    if n == 1:
        return [sequence + base for sequence in sequences for base in bases]
    else:
        sequence = [sequence + base for sequence in sequences for base in bases]
        return get_combinations(n - 1, sequence, bases)

def count_motif(motif, sequences_dataset):
    count = 0
    for sequence in sequences_dataset:
        count += sequence.count(motif)
    return count

def get_motif(motif_size, sequences_dataset):
    nucleotid_bases = ["A", "C", "G", "T"]
    combinations = get_combinations(motif_size, [""], nucleotid_bases)
    # get motif with the highest count
    max_counter = 0
    motif_winner = ""
    for motif_candidate in combinations:
        temp_counter = count_motif(motif_candidate, sequences_dataset)
        if temp_counter > max_counter:
            max_counter = temp_counter
            motif_winner = motif_candidate
    return motif_winner
def calculate_shannon_entrophy(motif):
    count = [motif.count('A'),motif.count('G'),motif.count('C'),motif.count('T')]
    entrophy = 0.0
    for c in count:
        if c != 0:
            entrophy -= (c/len(motif))*(math.log2(c/len(motif)))
    return entrophy

def filter_shannon(motif):
    eraser = True
    if calculate_shannon_entrophy(motif) > 1.79:
        print(calculate_shannon_entrophy(motif))
        eraser = False
    elif (len(motif) == 8 and calculate_shannon_entrophy(motif) >= 1.92):
        print(calculate_shannon_entrophy(motif))
        eraser = False
    return eraser

for size in [6, 8]:
    print(f"\nArter filter, motifs of size: {size}")
    dataset = create_dataset(50000)
    for i in range(10):
        dataset = [chain for chain in dataset if filter_shannon(get_motif(6,create_database())) ]
        print(f"Dataset size: {len(dataset)}, Motif: {get_motif(size, dataset)}")
