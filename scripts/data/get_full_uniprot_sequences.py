import os
import sys
import requests
from collections import Counter

if __name__ == "__main__":
    path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/no_duplicates/uniprot"

    fasta = os.path.join(path, "unique_uniprot_sequences.fasta")

    with open(fasta, "r") as f:
        ids = []
        for line in f:
            if line.startswith(">"):
                ids.append(line[1:].strip())
    
    # some don't have uniprot ids
    if "UNDEFINED" in ids:
        ids.remove("UNDEFINED")
    
    uniprot_stream = "https://rest.uniprot.org/uniprotkb/stream"

    out = {}
    tax_ids = {}
    species_counter = Counter()
    s = requests.Session()
    batch_size = 50
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i+batch_size]

        query = " OR ".join([f"accession:{uid}" for uid in batch])
        params = { "query": query, "format": "fasta" }

        print(f"Fetching {i}-{i+len(batch)} / {len(ids)}")

        r = s.get(uniprot_stream, params=params, timeout=60)
        r.raise_for_status()

        # parse fasta to dictionary
        lines = r.text.split("\n")

        # parse fasta to dictionary
        uid = None
        seq = []
        collecting = False
        for i, line in enumerate(lines):
            if i == len(lines)-1:
                # save last entry
                if collecting:
                    out[uid] = "".join(seq)

            if line.startswith(">"):
                # save previous, only not the case for first entry
                if collecting:
                    out[uid] = "".join(seq)
                    seq = []

                uid = line.strip().split("|")[1]
                collecting = True

                # extract species info
                split_one = line.split("OS=")[1].split(" ")[0].strip()
                split_two = line.split("OS=")[1].split(" ")[1].strip()
                species_counter[split_one + " " + split_two] += 1

                # extract tax id
                tax_id = line.split("OX=")[1].split(" ")[0].strip()
                tax_ids[split_one + " " + split_two] = tax_id

            else:
                seq.append(line.strip())

    # Missing ids in the output
    print("\n")
    missing = []
    for id in ids:
        if id not in out.keys():
            missing.append(id)
    print(f"Missing {len(missing)} / {len(ids)} uniprot ids:")
    print(",".join(missing))

    # write to fasta
    out_file = os.path.join(path, "full_uniprot_sequences.fasta")
    with open(out_file, "w") as f:
        for uid, seq in out.items():
            f.write(f">{uid}\n{seq}\n")
    
    out_species = os.path.join(path, "uniprot_species_distribution.tsv")
    with open(out_species, "w") as f:
        f.write("Species\tTax ID\tCount\n")
        for species, count in species_counter.most_common():
            f.write(f"{species}\t{tax_ids[species]}\t{count}\n")

