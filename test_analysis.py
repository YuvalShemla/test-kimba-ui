import pandas as pd
from load_data import load_night_data
from analysis import create_comparison_plot, FEATURES_TO_PLOT
import plotly.io as pio

def test_analysis():
    # Load the data
    print("Loading data...")
    df = load_night_data()
    
    if df is None:
        print("Failed to load data. Please check if the CSV file exists and is properly formatted.")
        return
    
    # Print basic information about the dataset
    print("\nDataset Overview:")
    print(f"Total Users: {len(df['user_id'].unique())}")
    print(f"Total Nights: {len(df)}")
    print("\nColumns in the dataset:")
    for col in df.columns:
        print(f"- {col}")
    
    # Test the analysis for each feature
    print("\nTesting analysis for each feature...")
    for feature, (display_name, unit) in FEATURES_TO_PLOT.items():
        print(f"\nAnalyzing {display_name}...")
        
        # Create the plot
        fig = create_comparison_plot(df, feature, display_name, unit, min_active_nights=5)
        
        if fig is not None:
            # Save the plot as HTML for interactive viewing
            output_file = f"test_output_{feature}.html"
            pio.write_html(fig, file=output_file)
            print(f"Plot saved as {output_file}")
            
            # Print some statistics about the feature
            active_data = df[df['diffuser_category'] == 'active'][feature]
            control_data = df[df['diffuser_category'] == 'control'][feature]
            
            print(f"\nStatistics for {display_name}:")
            print(f"Active nights mean: {active_data.mean():.2f} {unit}")
            print(f"Control nights mean: {control_data.mean():.2f} {unit}")
            print(f"Overall mean: {df[feature].mean():.2f} {unit}")
            print(f"Number of active nights: {len(active_data)}")
            print(f"Number of control nights: {len(control_data)}")
        else:
            print(f"Not enough data to create the plot for {display_name}")

if __name__ == "__main__":
    test_analysis() 