import numpy as np
import matplotlib.pyplot as plt
import os # To check if file exists

# --- Configuration ---
csv_filename = 'mopso_zdt1_objectives_en.csv'
plot_filename = 'mopso_zdt1_comparison_plot.png'

# --- Check if CSV file exists ---
if not os.path.exists(csv_filename):
    print(f"Error: The file '{csv_filename}' was not found.")
    print("Please make sure you have run the MOPSO code first to generate the CSV.")
else:
    try:
        # --- Read data from CSV ---
        # skiprows=1 to skip the header row ('f1,f2')
        # delimiter=',' specifies the separator
        mopso_objectives = np.loadtxt(csv_filename, delimiter=',', skiprows=1)

        # Extract f1 and f2 values
        mopso_f1 = mopso_objectives[:, 0]
        mopso_f2 = mopso_objectives[:, 1]

        # --- Calculate Theoretical Pareto Front for ZDT1 ---
        # The true Pareto front for ZDT1 is f2 = 1 - sqrt(f1) for f1 in [0, 1]
        theoretical_f1 = np.linspace(0, 1, 100) # 100 points for a smooth curve
        theoretical_f2 = 1 - np.sqrt(theoretical_f1)

        # --- Plotting ---
        plt.figure(figsize=(8, 6)) # Create a figure

        # Scatter plot for the MOPSO results
        plt.scatter(mopso_f1, mopso_f2, s=20, c='blue', marker='o', label='MOPSO Results')

        # Line plot for the theoretical Pareto front
        plt.plot(theoretical_f1, theoretical_f2, 'r--', linewidth=2, label='Theoretical Pareto Front (ZDT1)')

        # --- Customize Plot ---
        plt.title('MOPSO Results vs Theoretical Pareto Front for ZDT1')
        plt.xlabel('Objective 1 (f1)')
        plt.ylabel('Objective 2 (f2)')
        plt.legend() # Show the legend
        plt.grid(True) # Add a grid
        plt.xlim(left=0) # Start x-axis at 0
        plt.ylim(bottom=0) # Start y-axis at 0

        # --- Save or Show Plot ---
        try:
            plt.savefig(plot_filename) # Save the plot to a file
            print(f"Plot saved successfully as '{plot_filename}'")
        except Exception as e:
            print(f"Error saving plot: {e}")

        # plt.show() # Uncomment this line to display the plot interactively

    except ImportError:
        print("Matplotlib not found. Cannot create the plot.")
        print("Please install it using: pip install matplotlib")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")