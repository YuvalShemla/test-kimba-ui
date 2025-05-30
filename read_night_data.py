import pandas as pd

def read_night_data():
    try:
        # Read the CSV file
        df = pd.read_csv('user_night_data.csv')
        
        # Print the columns of the dataframe
        print("\nColumns in the dataset:")
        for col in df.columns:
            print(f"- {col}")
            
        return df
    
    except FileNotFoundError:
        print("Error: user_night_data.csv file not found in the current directory")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    df = read_night_data() 