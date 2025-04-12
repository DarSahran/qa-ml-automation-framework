# scripts/generate_data.py

import pandas as pd
import random
from faker import Faker

fake = Faker()
random.seed(42)

def generate_data(num_rows=5000, output_path="data/sample_data.csv"):
    data = []

    for i in range(1, num_rows + 1):
        customer_id = i
        age = random.randint(18, 65)
        country = random.choice(["India", "USA", "UK", "Germany", "Canada", "France"])
        signup_date = fake.date_between(start_date="-5y", end_date="today")
        churned = random.choices([0, 1], weights=[0.7, 0.3])[0]  # Simulate 30% churn

        data.append({
            "customer_id": customer_id,
            "age": age,
            "country": country,
            "signup_date": signup_date,
            "churned": churned
        })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✅ Generated {num_rows} rows at {output_path}")

if __name__ == "__main__":
    generate_data()
