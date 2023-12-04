import random
from datetime import datetime, timedelta

# Function to generate a random date
def random_date(start, end):
    return start + timedelta(
        seconds=random.randint(0, int((end - start).total_seconds())))

# Function to generate a random end date 2-4 days after the start date
def random_end_date(start_date):
    return start_date + timedelta(days=random.randint(2, 4))

# Start and end dates for generating the start date
start = datetime(2024, 1, 1)
end = datetime(2025, 12, 31)

# Generate 100 random start dates and corresponding end dates
dates =[]
for _ in range(1600):
    start_date = random_date(start, end)
    end_date = random_end_date(start_date)
    dates.append((start_date, end_date))

# Format dates and write to file
with open('dates.txt', 'w') as f:
    for start_date, end_date in dates:
        f.write(start_date.strftime('%Y-%m-%d') + ',' + end_date.strftime('%Y-%m-%d') + '\n')