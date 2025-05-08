import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def connect_to_database():
    """Connect to the SQLite database."""
    return sqlite3.connect('data/database.sqlite')

def get_table_names(conn):
    """Get all table names from the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [table[0] for table in cursor.fetchall()]

def explore_table(conn, table_name):
    """Get basic information about a table."""
    query = f"SELECT * FROM {table_name} LIMIT 5"
    df = pd.read_sql_query(query, conn)
    print(f"\nTable: {table_name}")
    print("\nFirst 5 rows:")
    print(df)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    return df

def main():
    # Connect to the database
    conn = connect_to_database()
    
    # Get all table names
    tables = get_table_names(conn)
    print("Available tables in the database:")
    for table in tables:
        print(f"- {table}")
    
    # Explore some key tables
    key_tables = ['Player', 'Match', 'Team', 'Player_Attributes']
    for table in key_tables:
        if table in tables:
            explore_table(conn, table)
    
    # Close the connection
    conn.close()

if __name__ == "__main__":
    main() 