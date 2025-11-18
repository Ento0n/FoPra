import os
import sys
import requests

def main():
    path = "/nfs/scratch/pinder/negative_dataset/my_repository/datasets/no_duplicates/uniprot"

    species_file = os.path.join(path, "uniprot_species_distribution.tsv")

    tax_ids = []
    with open(species_file, "r") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                tax_id = parts[1]
                tax_ids.append(tax_id)

    def get_lineage_for_taxid(taxid, ranks):
        url = f"https://rest.uniprot.org/taxonomy/{taxid}"
        r = requests.get(url, headers={"Accept": "application/json"}, timeout=60)
        r.raise_for_status()
        data = r.json()

        lineage_dict = {}
        lineage = data["lineage"]
        for dict in lineage:
            rank = dict["rank"]
            if rank.lower() == "clade":
                continue

            if rank not in ranks:
                print(f"Unknown rank: {rank} for taxid {taxid}")

            scientific_name = dict["scientificName"]

            # Since viruses do not have a domain, we set it here
            if scientific_name.lower() == "viruses":
                lineage_dict["domain"] = scientific_name

            lineage_dict[rank] = scientific_name
        
        if data["rank"].lower() == "species":
            lineage_dict["species"] = data["scientificName"]
        
        for rank in ranks:
            if rank not in lineage_dict.keys():
                lineage_dict[rank] = "NA"

        return {
            "taxon_id": data["taxonId"],
            "scientific_name": data["scientificName"],
        }, lineage_dict
    
    ranks = ["no rank", "domain", "kingdom", "phylum", "subphylum", "superclass", "class", "superorder", "order", "suborder", "infraorder", "parvorder", "superfamily", "family", "subfamily", "genus", "species"]

    infos = {}
    for tax_id in tax_ids:
        info, lineage_dict = get_lineage_for_taxid(tax_id, ranks)
        infos[tax_id] = (info["scientific_name"], lineage_dict)
    
    with open(os.path.join(path, "uniprot_lineage.tsv"), "w") as out_f:
        out_f.write("tax_id\tscientific_name\t" + "\t".join(ranks) + "\n")
        for tax_id, (sci_name, lineage) in infos.items():
            out_f.write(f"{tax_id}\t{sci_name}\t" + "\t".join([lineage[rank] for rank in ranks]) + "\n")


if __name__ == "__main__":
    main()