import pandas as pd

# Load the CSV file
file_path = "train.csv"
df = pd.read_csv(file_path)

# Check for null values in 'Postal Code' column
null_postal_code_count = df['Postal Code'].isnull().sum()
print(f"Number of null values in 'Postal Code': {null_postal_code_count}")

# Convert 'Postal Code' to string type first to avoid dtype conflicts
df['Postal Code'] = df['Postal Code'].astype('string')

# Fill null postal codes with a placeholder (e.g., 'Unknown')
df['Postal Code'] = df['Postal Code'].fillna('Unknown')

# Optional: Strip whitespace from postal codes
df['Postal Code'] = df['Postal Code'].str.strip()

# Save the cleansed data to a new CSV file
cleansed_file_path = "train_cleansed.csv"
df.to_csv(cleansed_file_path, index=False)

print(f"Cleansed data saved to '{cleansed_file_path}'")
