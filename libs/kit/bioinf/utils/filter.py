from Bio import PDB

from kit.bioinf.proteins import Protein, ProteinType
from kit.bioinf.pdb import download_structure, get_unmodelled_residues, has_small_molecules


def filter_pdbs(pdb_ids, pdb_dir_path, 
                max_unmodelled=5, exclude_small_molecules=True, exclude_complexes=False,
                server=r'https://files.rcsb.org'):
    # Function to filter out PDBs with unmodeled parts or small molecules

    pdbl = PDB.PDBList()
    valid_pdbs = []
    
    failed_downloads = []
    for pdb_id in pdb_ids:
        try:
            pdb_file_path = download_structure(pdb_id, pdb_dir_path, server=server)
            # Check for unmodeled parts and small molecules
            unmodelled = get_unmodelled_residues(pdb_file_path)
            exclude = False
            if len(unmodelled) > max_unmodelled:
                exclude = True       
            elif exclude_small_molecules and has_small_molecules(pdb_file_path):
                exclude = True
            elif exclude_complexes:
                protein = Protein.from_pdb(pdb_file_path)
                if protein.get_protein_type() == ProteinType.COMPLEX:
                    exclude = True

            if not exclude:
                valid_pdbs.append(pdb_id)
        except Exception as e:
            failed_downloads.append(pdb_id)
            continue

    print(f"Failed downloads: {len(failed_downloads)}/{len(pdb_ids)}")

    return valid_pdbs