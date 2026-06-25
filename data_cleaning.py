import pandas as pd

def clean_complaints_data(file_path):
    print("Loading data, this might take a few seconds...")
    
    # 1. Load raw data
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Data loaded successfully! Total records: {len(df)}")
    except FileNotFoundError:
        print("Error: Data file not found. Please check the file path.")
        return None

    # 2. Select relevant columns for our RAG project
    columns_to_keep = [
        'Date received', 
        'Product', 
        'Issue', 
        'Company', 
        'State', 
        'Consumer complaint narrative' # The most important column for NLP/RAG
    ]
    
    # Ensure columns exist before filtering
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns]

    # 3. Data Cleaning
    # Drop rows where the complaint text is missing (crucial for our AI model)
    df = df.dropna(subset=['Consumer complaint narrative'])
    print(f"Records remaining after dropping empty complaints: {len(df)}")

    # 4. Feature Engineering
    # Convert 'Date received' to datetime format for easier analysis in Tableau
    if 'Date received' in df.columns:
        df['Date received'] = pd.to_datetime(df['Date received'])

    # Calculate the length of each complaint (number of words)
    df['Complaint_Length'] = df['Consumer complaint narrative'].apply(lambda x: len(str(x).split()))

    # 5. Sampling for Prototype Speed
    # Take the latest 5000 records to build and test the RAG model quickly
    df_sample = df.sort_values(by='Date received', ascending=False).head(5000)
    print(f"Sampled {len(df_sample)} records for the initial prototype.")

    # 6. Export cleaned data
    output_filename = 'cleaned_complaints_data.csv'
    df_sample.to_csv(output_filename, index=False)
    print(f"Cleaning complete! Data saved to: {output_filename}")
    
    return df_sample

if __name__ == "__main__":
    # Define the input file name (must be in the same directory)
    input_file = 'complaints.csv'
    
    # Run the cleaning function
    cleaned_df = clean_complaints_data(input_file)
    
    if cleaned_df is not None:
        print("\nQuick look at the first 3 cleaned records:")
        print(cleaned_df.head(3))