
import os

import psycopg2
import psycopg2.extras

from colorama import Fore, Style


def get_latest_trained_row(experiment):

    print(Fore.BLUE + "\nRetrieve last trained row from mlflow db..." + Style.RESET_ALL)

    # get latest trained row
    mlflow_query = f"""
        SELECT
            pa.key AS param_key,
            pa.value AS param_value
        FROM runs ru
        JOIN experiments ex ON ex.experiment_id = ru.experiment_id
        JOIN params pa ON pa.run_uuid = ru.run_uuid
        WHERE ex.name = '{experiment}'
        AND pa.key IN ('first_row', 'row_count')
        ORDER BY ru.end_time DESC
        LIMIT 2;
        """

    tracking_db_uri = os.environ.get("MLFLOW_TRACKING_DB")

    conn = psycopg2.connect(tracking_db_uri)

    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(mlflow_query)

    # retrieve content
    first_row = 0
    row_count = 0

    results = cur.fetchall()

    for row in results:

        row_key = row["param_key"]
        row_value = row["param_value"]

        if row_key == "first_row":
            first_row = int(row_value)
        elif row_key == "row_count":
            row_count = int(row_value)

    next_row = first_row + row_count

    print(f"\n✅ last trained rows: row {next_row}")

    return next_row
