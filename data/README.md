# Data

Place the dataset at data/customer_data.csv.

Expected fields:
customer_id, credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary, churn.

The baseline experiment used 10,000 rows with an observed churn rate of 20.37%.

The dataset in the earlier forked project is the source of the baseline schema/results. This repository rebuilds the modeling and serving layers around that baseline. Confirm the dataset's original provenance and reuse terms before using it outside an educational portfolio.

Train locally with:
python -m src.train --data data/customer_data.csv
