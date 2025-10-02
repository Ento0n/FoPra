import os
import requests

if __name__ == "__main__":
    path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/deleak_uniprot/deleak_cdhit/unique_sequences_uniprot_id"

    fasta = os.path.join(path, "unique_uniprot_ids.fasta")

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
        uid = None
        seq = []
        for line in lines:
            if line.startswith(">"):
                if uid not in out.keys() and uid is not None:
                    out[uid] = "".join(seq)
                    seq = []

                uid = line[1:].split("|")[1]
            else:
                seq.append(line.strip())
    
    out_file = os.path.join(path, "full_uniprot_sequences.fasta")
    with open(out_file, "w") as f:
        for uid, seq in out.items():
            f.write(f">{uid}\n{seq}\n")

