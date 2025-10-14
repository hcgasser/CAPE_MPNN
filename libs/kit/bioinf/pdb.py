"""This module contains functions for working with PDB files"""

import os
import sys
import requests
from Bio import PDB
import py3Dmol

from kit.path import join
from kit.log import log_info
from kit.bioinf.utils import structure_to_seq


def cif_to_pdb(cif_file_path, pdb_file_path):
    """Converts a CIF file to a PDB file"""

    # Create a PDB parser and structure object
    parser = PDB.MMCIFParser()
    structure = parser.get_structure("structure_id", cif_file_path)

    # Create a PDBIO object for saving the structure in PDB format
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(pdb_file_path)


def pdb_to_seqs(file_path, return_full=True, gaps='-', aa3_replace=None, aa_ids=[' ']):
    """reads a PDB file and returns the sequence"""
    return structure_file_to_seqs(file_path, return_full=return_full, gaps=gaps, aa3_replace=aa3_replace, aa_ids=aa_ids)


def structure_file_to_seqs(file_path, return_full=True, gaps='-', aa3_replace=None, aa_ids=[' ']):
    if file_path.endswith(".mmCif"):
        parser = PDB.MMCIFParser()
    elif file_path.endswith(".pdb"):
        parser = PDB.PDBParser()
    structure = parser.get_structure("X", file_path)
    return structure_to_seq(structure, return_full=return_full, gaps=gaps, aa3_replace=aa3_replace, aa_ids=aa_ids)


def download_pdb(pdb_id, output_dir, overwrite=False, server=r'https://files.rcsb.org'):
    return download_structure(pdb_id, output_dir, overwrite=overwrite, file_format='pdb', server=server)


def download_AF_EBI(uniprot_id, output_file_path):
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(output_file_path, 'wb') as file:
            file.write(response.content)
    else:
        raise Exception(f"Download {uniprot_id} failed")


def download_structure(protein_id, output_dir, overwrite=False, file_format="pdb", source='pdb', server=r'https://files.rcsb.org'):
    """Downloads a PDB file from the RCSB PDB database"""
    pdbl = PDB.PDBList(server=server)

    output_file_path = os.path.join(output_dir, f"{protein_id}.{file_format}")
    if overwrite or not os.path.exists(output_file_path):
        if source == 'pdb':
            file_name = pdbl.retrieve_pdb_file(protein_id, pdir=output_dir, file_format=file_format)
            output_file_path = join(output_dir, file_name)
        elif source == 'AF_EBI':
            output_file_path = join(output_dir, f"{protein_id}.{file_format}")
            download_AF_EBI(protein_id, output_file_path)

    if os.path.exists(output_file_path):
        pdb_file_path = join(output_dir, f"{protein_id}.{file_format}")
        os.rename(output_file_path, pdb_file_path)
    else:
        raise Exception(f"Download {protein_id} failed")

    return pdb_file_path


def get_similar_structures(pdb_id):
    # see: https://search.rcsb.org/
    r = {}
    url = f'https://search.rcsb.org/rcsbsearch/v2/query'

    payload = {
      "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "terminal",
                "service": "structure",
                "parameters": {
                  "value": {
                    "entry_id": pdb_id,
                    "assembly_id": "1"
                  },
                  "operator": "strict_shape_match"
                }
            }
        ]
      },
      "request_options": {
        "return_all_hits": True
      },
      "return_type": "entry"
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        results = response.json()
        for entry in results['result_set']:
            r[entry['identifier']] = entry['score']
    else:
        print(f"Failed to retrieve data: {response.status_code}")
    return r


def view_structure(pdb_file_path, res_colors=None, output_file_path=None, width=800, height=800):
    """Displays a 3D structure of a protein"""
    print(pdb_file_path)

    with open(pdb_file_path) as ifile:
        system = "".join([x for x in ifile])
   
    view = py3Dmol.view(width=width, height=height)
    view.addModelsAsFrames(system)

    colors = [
        '#22FFFF',
        '#44BBBB',
        '#669999',
        '#887777',
        '#AA5555',
        '#CC3333',
        '#EE1111',
    ]

    i = 0
    for line in system.split("\n"):
        split = line.split()

        if len(split) == 0 or split[0] != "ATOM":
            continue

        if res_colors is not None:
            resid = int(split[5])
            color = res_colors[resid-1]
            view.setStyle({'model': -1, 'serial': i+1}, {"sphere": {'color': color}})
        else:
            view.setStyle({'model': -1, 'serial': i+1}, {"cartoon": {}})
        i += 1

    view.zoomTo()
    view.show()
    view.render(filename=output_file_path)


def get_unmodelled_residues(pdb_file_path):
    # Function to check if there are missing residues in the PDB file

    unmodelled = []
    section_465 = False
    section_465_end = False
    with open(pdb_file_path, 'r') as file:
        for line in file:
            if line.startswith("REMARK 465"):
                if line.startswith("REMARK 465   M RES C SSSEQI"):
                    section_465 = True
                elif section_465:
                    parts = line.split()
                    if len(parts) == 5:
                        unmodelled.append(parts[4])
                    else:
                        section_465_end = True
                        section_465 = False
                else:
                    if section_465_end:
                        raise Exception("Error in REMARK 465")

    return unmodelled


def has_small_molecules(pdb_file_path, accept_water=True):
    # Function to check if there are small molecules in the PDB file

    small_molecules = False
    
    with open(pdb_file_path, 'r') as file:
        for line in file:
            # HETATM entries describe non-standard residues or ligands (small molecules)
            if line.startswith("HETATM") and (not accept_water or line[17:20].strip() != 'HOH'):
                small_molecules = True
                break
    return small_molecules


def includes_dna(pdb_file_path):
    return any(' DA ' in line or ' DT ' in line or ' DG ' in line or ' DC ' in line for line in open(pdb_file_path))


def check_xray(pdb_file_path):
    with open(pdb_file_path, 'r') as file:
        for line in file:
            if line.startswith('EXPDTA'):
                if 'X-RAY DIFFRACTION' in line.upper():
                    return True
                else:
                    return False
    return None


def get_protein_name_organism(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    header = structure.header
    protein_name = header.get('name', 'Unknown')

    source = header.get('source', {})
    for mol_id, mol_info in source.items():
        if 'organism_scientific' in mol_info:
            organism = mol_info['organism_scientific']
    
    return protein_name, organism


def structure_to_pdb(structure, pdf_file_path):
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(pdf_file_path)


def reduce_pdb_to_CA(pdb_file_path, output_file_path):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file_path)

    atoms_to_delete = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                atoms_to_delete[residue] = []
                for atom in residue:
                    atom_id = atom.get_id()
                    if atom_id != 'CA':
                        atoms_to_delete[residue].append(atom_id)

    # cannot be deleted inside the loop
    for residue, atom_ids in atoms_to_delete.items():
        for atom_id in atom_ids:
            residue.detach_child(atom_id)

    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_file_path)
