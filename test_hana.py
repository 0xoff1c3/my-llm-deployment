from hdbcli import dbapi

HANA_HOST = "73a8a9e6-3c93-49a4-8dd3-c5dfb09ea621.hana.trial-us10.hanacloud.ondemand.com"
HANA_PORT = 443
HANA_USER = "DBADMIN"         # or your custom DB user
HANA_PASSWORD = "Hana@123"  

try:
    conn = dbapi.connect(
        address=HANA_HOST,
        port=HANA_PORT,
        user=HANA_USER,
        password=HANA_PASSWORD
    )

    cursor = conn.cursor()
    cursor.execute("SELECT TABLE_NAME FROM TABLES WHERE SCHEMA_NAME = CURRENT_SCHEMA")
    tables = cursor.fetchall()
    print("Tables in current schema:")
    for t in tables:
        print(" -", t[0])

    # If you want to preview data from a specific table, replace TABLE_NAME_HERE
    if tables:
        first_table = tables[0][0]
        print(f"\n Previewing 5 rows from {first_table}:")
        cursor.execute(f"SELECT * FROM {first_table} LIMIT 15")
        rows = cursor.fetchall()
        for row in rows:
            print(row)

    cursor.close()
    conn.close()

except Exception as e:
    print("Error:", e)