import pandas as pd
import os

def clean_data():
    input_dir = "bronze_layer"
    output_dir = "silver_layer"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Silver Layer Transformation Shuru...")

    # Bronze folder mein jitni bhi CSVs hain sab utha lega
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_dir, filename)
            print(f"Cleaning: {filename}")
            
            # Load Data
            df = pd.read_csv(file_path)
            
            # Remove duplicates
            df.drop_duplicates(inplace=True)

            # Fill missing values based on column type
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna("Unknown")
            
            # 2. Standardize column names (lowercase, no spaces)
            df.columns = [col.lower().replace(" ", "_") for col in df.columns]
            
            # 3. Save to Silver Layer
            save_path = os.path.join(output_dir, filename.replace(".csv", "_silver.csv"))
            df.to_csv(save_path, index=False)
            print(f"Saved: {save_path}")

    print("\n✅ Silver Layer Done! Data ab cleaned hai.")

if __name__ == "__main__":
    clean_data()