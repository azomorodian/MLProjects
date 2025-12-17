import os
print("--- بررسی متغیر محیطی ---")
print("GOOGLE_APPLICATION_CREDENTIALS:", os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
print("--- پایان بررسی ---")
BILLING_PROJECT_ID = "artinpycharm"
KAGGLE_PROJECT_ID = "bigquery-public-data"

from google.cloud import bigquery
client = bigquery.Client(project=BILLING_PROJECT_ID)
dataset_ref = client.dataset("openaq", project=KAGGLE_PROJECT_ID)

dataset = client.get_dataset(dataset_ref)
tables = list(client.list_tables(dataset))
for table in tables:
    print(table.table_id)
table_ref = dataset_ref.table("global_air_quality")
table = client.get_table(table_ref)
client.list_rows(table, max_results=5).to_dataframe()
query = """
        SELECT city
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US'
        """
query_job = client.query(query)
us_cities = query_job.to_dataframe()
print(us_cities.city.value_counts().head())
query = """
        SELECT city, country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US'
        """
query = """
        SELECT *
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US'
        """

query = """
        SELECT score, title
        FROM `bigquery-public-data.hacker_news.full`
        WHERE type = "job" 
        """
# Create a QueryJobConfig object to estimate size of query without running it
dry_run_config = bigquery.QueryJobConfig(dry_run=True)

# API request - dry run query to estimate costs
dry_run_query_job = client.query(query, job_config=dry_run_config)

print("This query will process {} bytes.".format(dry_run_query_job.total_bytes_processed))