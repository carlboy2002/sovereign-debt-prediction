from google.cloud import bigquery

client = bigquery.Client()
print(client.query("SELECT 1").to_dataframe())
