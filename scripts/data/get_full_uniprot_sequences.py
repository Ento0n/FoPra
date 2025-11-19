import os
import requests
from collections import Counter

if __name__ == "__main__":
    path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/no_duplicates/deleak_cdhit/sunburst_data"

    print(f"Getting information like full Sequences, Tax ID and Organism for uniprot ids in {path}\n")

    fasta = os.path.join(path, "unique_uniprot_sequences.fasta")

    with open(fasta, "r") as f:
        ids = []
        id_to_count = {}
        for line in f:
            if line.startswith(">"):
                id = line[1:].split("|")[0].strip()
                count = int(line[1:].split("|")[1].strip().split("#=")[1].strip())
                ids.append(id)
                id_to_count[id] = count
    
    # some don't have uniprot ids
    if "UNDEFINED" in ids:
        ids.remove("UNDEFINED")
    
    uniprot_stream = "https://rest.uniprot.org/uniprotkb/stream"

    out = {}
    tax_ids = {}
    species_uniq_counter = Counter()
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
                species_uniq_counter[split_one + " " + split_two] += 1

                # extract tax id
                tax_id = line.split("OX=")[1].split(" ")[0].strip()
                tax_ids[split_one + " " + split_two] = tax_id

                # add tax id to header
                uid = f"{uid}|OX={tax_id}|#={id_to_count[uid]}"

            else:
                seq.append(line.strip())

    # Missing ids in the output
    print("\n")
    missing = []
    out_ids = [key.split("|")[0].strip() for key in out.keys()]
    for id in ids:
        if id not in out_ids:
            missing.append(id)
    print(f"Missing {len(missing)} / {len(ids)} uniprot ids:")
    print(",".join(missing))

    # write to fasta
    taxid_counter = Counter()
    out_file = os.path.join(path, "full_uniprot_sequences.fasta")
    with open(out_file, "w") as f:
        for header, seq in out.items():
            f.write(f">{header}\n{seq}\n")

            # fill counter
            tax_id = header.split("|OX=")[1].split("|")[0].strip()
            count = int(header.split("|#=")[1].strip())
            taxid_counter[tax_id] += count
    
    # write species distribution
    out_species = os.path.join(path, "uniprot_species_distribution.tsv")
    with open(out_species, "w") as f:
        f.write("Species\tTax ID\tCount_uniq\tCount_total\n")
        for species, count in species_uniq_counter.most_common():  # most_common gives sorted order, doesn't reduce size
            f.write(f"{species}\t{tax_ids[species]}\t{count}\t{taxid_counter[tax_ids[species]]}\n")

