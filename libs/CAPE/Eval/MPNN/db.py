"""
The database class used during the CAPE-MPNN evaluation.
It manages the sequences, their visibility and structural similarity
"""

import os
from tqdm.auto import tqdm

from kit.data.db import BasicDB
from kit.loch.utils import get_seq_hash
from kit.data.trees import PrefixTree
from kit.hashes import str_to_hash
from kit.bioinf.fasta import read_fasta
from kit.bioinf import AA1_FULL


class CapeMpnnDB(BasicDB):
    proteomes = {}
    proteome_trees = {}

    def __init__(self, database_path, mhc_1_predictor):
        """ initializes the database

        if the database file does not exist, it will be created and the tables will be initialized

        Args:
            database_path (str): path to the database file
            mhc_1_predictor (MHC1Predictor): MHC1Predictor object
        """

        create_db = False
        if not os.path.exists(database_path):
            create_db = True
        super().__init__(database_path)
        self.mhc_1_predictor = mhc_1_predictor

        if create_db:
            self.create_database(all=True)

    def create_database(self, all=False):
        """ creates the database tables 
        
        if all is False, the visibility_mhc_1 table will not be created
        """

        cursor = self.cursor

        # Retrieve table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table';")
        tables = cursor.fetchall()

        # Drop tables
        for table in tables:
            table_name = table[0]
            if all or table_name not in ['visibility_mhc_1']:
                cursor.execute(f"DROP TABLE IF EXISTS {table_name};")

        cursor.execute("""
            CREATE TABLE sequences (
                seq_hash TEXT PRIMARY KEY,
                seq TEXT,
                complete BOOLEAN,
                UNIQUE(seq_hash)
            )""")

        cursor.execute("""
            CREATE TABLE lists (
                grp TEXT,
                source TEXT,
                id INTEGER,
                designed_positions TEXT,
                trial INTEGER DEFAULT 1,
                seq_hash TEXT,
                UNIQUE(grp, source, id, designed_positions, trial),
                FOREIGN KEY (seq_hash) REFERENCES sequences(seq_hash)
            )""")

        cursor.execute("""
            CREATE TABLE tm_align (
                seq_hash_1 TEXT,
                seq_hash_2 TEXT,
                score NUMBER,
                alignment_length NUMBER,
                rmsd NUMBER,
                identical NUMBER,
                len_chain_1 INTEGER,
                len_chain_2 INTEGER,
                complete BOOLEAN,
                UNIQUE(seq_hash_1, seq_hash_2),
                FOREIGN KEY (seq_hash_1) REFERENCES sequences(seq_hash),
                FOREIGN KEY (seq_hash_2) REFERENCES sequences(seq_hash)
            )""")

        if all:
            cursor.execute("""
                CREATE TABLE visibility_mhc_1 (
                    seq_hash TEXT,
                    predictor_hash TEXT,
                    proteome_hash TEXT,
                    immuno_setup_mhc_1 TEXT,
                    visibility INTEGER,
                    UNIQUE(seq_hash, predictor_hash, proteome_hash, immuno_setup_mhc_1),
                    FOREIGN KEY (seq_hash) REFERENCES sequences(seq_hash)
                )""")

    def add_seq(self, seq):
        """ adds a sequence to the database """

        seq_hash = get_seq_hash(seq)

        sql = """
            INSERT OR IGNORE INTO sequences (seq_hash, seq, complete)
            VALUES (?, ?, ?)
        """
        self.cursor.execute(sql, (seq_hash, seq, "X" not in seq))
        return seq_hash

    def add_seq_to_list(self, group, source, id, designed_positions, trial, seq_hash):
        """ adds a sequence to a list
        
        a list has
        - a group... e.g. general:test
        - a source... e.g. data, v_048_02, a checkpoint id, ....
        - an id... 
            for specific sequences the pdb id
            for general sequences a continuous number (1, 2, 3, ...) to distinguish them
        - designed_positions... saves which positions were generated (e.g. 'all', ...)
        - trial... the trial number
        
        Args:
            group (str): list's group name
            source (str): list's source name
            id (int): list's id
            designed_positions (str): list's designed positions
            trial (int): list's trial number
            seq_hash (str): sequence's hash
        """

        sql = """
            INSERT OR IGNORE INTO lists (grp, source, id, designed_positions, trial, seq_hash)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        self.cursor.execute(sql, (group, source, id, designed_positions, trial, seq_hash))
        self.conn.commit()

    def get_list(self, group, source, designed_positions_name, return_seq=False):
        """ retrieves the sequences of a list specified by the function's arguments"""

        if not return_seq:
            sql = f"""
                SELECT l.grp, l.source, l.id, l.trial, l.seq_hash
                FROM lists l
                WHERE l.grp = '{group}' 
                    AND l.source = '{source}' 
                    AND l.designed_positions = '{designed_positions_name}'
                ORDER BY l.grp, l.source, l.id, l.trial, l.seq_hash
            """
        else:
            sql = f"""
                SELECT l.grp, l.source, l.id, l.trial, l.seq_hash, s.seq
                FROM lists l LEFT JOIN sequences s ON l.seq_hash = s.seq_hash
                WHERE l.grp = '{group}' 
                    AND l.source = '{source}' 
                    AND l.designed_positions = '{designed_positions_name}'
                ORDER BY l.grp, l.source, l.id, l.trial, l.seq_hash
            """
        return self.sql_to_df(sql)

    def add_tm_align(self, seq_hash_1, seq_hash_2, score, alignment_length, rmsd, identical, 
            len_chain_1, len_chain_2):
        """ adds the TMalign results between two sequences to the database """

        sql = """
            INSERT OR IGNORE INTO tm_align (seq_hash_1, seq_hash_2, score, alignment_length, rmsd, identical, len_chain_1, len_chain_2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.cursor.execute(sql, 
            (seq_hash_1, seq_hash_2, score, alignment_length, rmsd, identical, 
            len_chain_1, len_chain_2)
        )
        self.conn.commit()

    def get_tm_align(self, seq_hash_1, seq_hash_2):
        """ retrieves the TMalign results between two sequences """

        sql = f"""
            SELECT score, alignment_length, rmsd, identical, len_chain_1, len_chain_2
            FROM tm_align
            WHERE seq_hash_1 = '{seq_hash_1}' AND seq_hash_2 = '{seq_hash_2}'
        """
        return self.sql_to_df(sql)

    def get_by_pdb(self, source, pdb, designed_positions):
        """ returns all the sequences for a specific pdb id coming from a specific source """

        sql = f"""
            SELECT l.seq_hash, s.seq
            FROM lists l LEFT JOIN sequences s ON l.seq_hash = s.seq_hash
            WHERE l.id = '{pdb}' AND l.source = '{source}' 
        """
        if designed_positions is not None:
            sql += f" AND l.designed_positions = '{designed_positions}'"
        return self.sql_to_df(sql)
            
    def add_visibility_mhc_1_missing(self, immuno_setup_mhc_1, 
            seq_hashes=None, proteome_hash=None, mhc_1_predictor=None):
        """ uses the MHC1Predictor to predict the visibility of the sequences in the database 
        which have not been predicted yet"""

        mhc_1_predictor = self.mhc_1_predictor if mhc_1_predictor is None else mhc_1_predictor
        predictor_hash = mhc_1_predictor.get_predictor_hash()
        alleles = mhc_1_predictor.resolve_alleles(immuno_setup_mhc_1)

        sql_proteome =  (
            f" v.proteome_hash = '{proteome_hash}' " 
            if proteome_hash is not None 
            else ' v.proteome_hash IS NULL '
        )
        sql = f"""
            SELECT DISTINCT s.seq_hash, s.seq
            FROM sequences s LEFT JOIN (
                SELECT v.seq_hash FROM visibility_mhc_1 v
                WHERE v.predictor_hash = '{predictor_hash}' 
                    AND v.immuno_setup_mhc_1 = '{immuno_setup_mhc_1}' 
                    AND {sql_proteome}
            ) vv ON s.seq_hash = vv.seq_hash
            WHERE vv.seq_hash IS NULL
        """
        df = self.sql_to_df(sql)
        rows_to_predict = []

        print(f"Predicting: {len(df) if seq_hashes is None else len(seq_hashes)} sequences")
        for _, row in df.iterrows():
            if seq_hashes is None or row.seq_hash in seq_hashes:
                res = self.get_visibility_mhc_1(
                    immuno_setup_mhc_1, 
                    row.seq_hash,  
                    proteome_hash=proteome_hash
                )
                if res is None:
                    rows_to_predict.append(row)
                    for _seq in row.seq.split('/'):
                        mhc_1_predictor.queue_seq(_seq, alleles)

        mhc_1_predictor.predict_missing_peptides()
        proteome_tree = None if proteome_hash is None else CapeMpnnDB.proteome_trees[proteome_hash]

        for row in tqdm(rows_to_predict):
            seqs = row.seq.split("/")

            visibility = 0
            for seq in seqs:
                presented = mhc_1_predictor.seq_presented(seq, immuno_setup_mhc_1)
                presented = set([(p[0], p[3]) for p in presented])
                if proteome_tree is not None:
                    presented = set([p for p in presented if not proteome_tree.has_kmer(p[0])])
                visibility += len(presented)

            sql = """
                INSERT OR IGNORE INTO visibility_mhc_1 (seq_hash, predictor_hash, proteome_hash, immuno_setup_mhc_1, visibility)
                VALUES (?, ?, ?, ?, ?)
            """
            self.cursor.execute(sql, 
                (row.seq_hash, predictor_hash, proteome_hash, immuno_setup_mhc_1, visibility)
            )
            self.conn.commit()

    def get_visibility_mhc_1(self, immuno_setup_mhc_1, seq_hash, 
            predictor_hash=None, proteome_hash=None):
        if predictor_hash is None:
            predictor_hash = self.mhc_1_predictor.get_predictor_hash()
        sql = f"""
            SELECT v.visibility FROM visibility_mhc_1 v
            WHERE v.predictor_hash = '{predictor_hash}' 
                AND v.immuno_setup_mhc_1 = '{immuno_setup_mhc_1}' 
                AND v.seq_hash = '{seq_hash}'
        """
        if proteome_hash is not None:
            sql += f""" AND v.proteome_hash = '{proteome_hash}' """
        df = self.sql_to_df(sql)
        if len(df) != 1:
            return None
        return df.iloc[0].visibility
    
    def get_seq(self, seq_hash):
        sql = f"SELECT * FROM sequences WHERE seq_hash = '{seq_hash}'"
        df = self.sql_to_df(sql)
        if len(df) != 1:
            return None
        return df.iloc[0]['seq']

    @classmethod
    def load_proteome(cls, proteome_file_path):
        proteome_hash = str_to_hash(os.path.basename(proteome_file_path), truncate=5)
        proteome = read_fasta(proteome_file_path, stop_token=False, evaluate=False, return_df=True)

        PrefixTree.set_alphabet(AA1_FULL)
        proteome_tree = PrefixTree()

        cls.proteomes[proteome_hash] = proteome
        cls.proteome_trees[proteome_hash] = proteome_tree

        wrong_seqs = []
        for seq, _ in tqdm(proteome.iterrows()):
            if all([(s in AA1_FULL) for s in seq]):
                proteome_tree.add_seq(seq, 10)
            else:
                wrong_seqs.append(seq)
        
        return proteome_hash
        