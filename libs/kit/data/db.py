""" Database class to access SQLite3 """

import time
import sqlite3
import pandas as pd


class BasicDB:
    """Basic database class to access SQLite3

    :param path: str - Path to the database file
    :param connect: boolean - Whether to connect to the database immediately
    """

    def __init__(self, path, connect=True):
        self.path = path
        self.conn = None
        self.cursor = None
        if connect:
            self.connect()

    def connect(self):
        """Connect to the database. If the database is locked,
        wait 10 minutes and try again."""

        while True:
            try:
                self.conn = sqlite3.connect(self.path)
                self.conn.row_factory = sqlite3.Row
                self.cursor = self.conn.cursor()
                break  # Database is no longer locked
            except sqlite3.OperationalError:
                print("Database is locked. Waiting...")
                time.sleep(10 * 60)  # Wait for 10 minutes before trying again

    def close(self):
        """Close the database connection"""

        self.conn.close()

    def sql_to_df(self, sql):
        """Execute SQL and return the result as a pandas dataframe"""
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        return pd.DataFrame(results, columns=[d[0] for d in self.cursor.description])

    def exists_table(self, table):
        """Check if a table exists in the database"""

        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        return len(self.cursor.fetchall()) > 0

    def exists_column(self, table, column, add=None):
        """Check if a column exists in a table. If add is not None,
        add the column to the table. In this case, add might
        be a string with the column type.

        :param table: str - Name of the table
        :param column: str - Name of the column
        :param add: str - Column type (e.g. "INTEGER DEFAULT NULL")
        """

        self.cursor.execute(f"PRAGMA table_info({table})")
        columns = [column[1] for column in self.cursor.fetchall()]
        if column not in columns:
            if add is not None:
                self.cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {add}")
            return False
        return True
