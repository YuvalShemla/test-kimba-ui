import pandas as pd

def load_night_data():
    """
    Load and prepare the night data from the new CSV file
    """
    try:
        df = pd.read_csv('users_night_stats_df.csv')
        df['user_id'] = df['user_id'].astype(str)
        return df
    except FileNotFoundError:
        print("Error: users_night_stats_df.csv file not found in the current directory")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def get_active_users(df, min_active_nights=5):
    """
    Get list of users with minimum number of active nights
    """
    active_nights = df[df['diffuser_category'] == 'active'].groupby('user_id').size()
    return active_nights[active_nights >= min_active_nights].index.tolist() 