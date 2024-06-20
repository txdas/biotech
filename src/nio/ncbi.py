from Bio import Entrez, SeqIO


def get_species_info(genbank_ids):
    Entrez.email = "your.email@example.com"  # 请使用你的电子邮件地址
    species_info = {}
    # 将GenBank ID列表转换为逗号分隔的字符串
    ids_str = ",".join(genbank_ids)
    try:
        handle = Entrez.efetch(db="nucleotide", id=ids_str, rettype="gb", retmode="text")
        records = SeqIO.parse(handle, "genbank")
        for record in records:
            organism = record.annotations.get('organism', 'Unknown')
            taxonomy = record.annotations.get('taxonomy',["Unknown"])[-1]
            species_info[record.id] = {"genbank":record.id, "organism":organism, "taxonomy":taxonomy}
        handle.close()
    except Exception as e:
        for genbank_id in genbank_ids:
            species_info[genbank_id] = {}
    return species_info
# 示例使用
genbank_id = ["NC_040548.1","NC_040550.1"]
species = get_species_info(genbank_id)
print(f"The species information for GenBank ID {genbank_id} is: {species}")