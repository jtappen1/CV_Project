from matplotlib import pyplot as plt
import pandas as pd


def plot_probability_differences():
    # Load the CSV file into a DataFrame
    file_path = "/Users/jtappen/Projects/cv_project/guitar_neck_detection/probabilities_final.csv"
    df = pd.read_csv(file_path)

    column_sums = df.sum()

    column_std = df.std()

    plt.figure(figsize=(10, 6))

    # X-axis will be the column names
    x = df.columns

    # Plot the sum of each column
    plt.plot(x, column_sums, label='Sum of Values', marker='o', color='b')

    # Fill between to show the standard deviation shadow
    plt.fill_between(x, column_sums - column_std, column_sums + column_std, color='b', alpha=0.2, label='±1 Std Dev')

    # Adding labels and title
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.title('Sum of Each Column with Standard Deviation Shadow')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate column labels if necessary
    plt.tight_layout()

    # Show the plot
    plt.show()

    


    # Coordinate Plot
    # plt.figure(figsize=(10, 6))

    # # X-axis will be the column names
    # x = df.columns

    # # Iterate over each row and plot its values as a line
    # for index, row in df.iterrows():
    #     plt.plot(x, row, marker='o', label=f'Question {index+1}')  # 'Question' labels

    # # Add labels and title
    # plt.xlabel('Columns')
    # plt.ylabel('Values')
    # plt.title('CSV Rows Comparison by Column')
    # plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # Legend outside the plot
    # plt.grid(True)
    # plt.xticks(rotation=45)  # Rotate column labels if necessary for better readability
    # plt.tight_layout()

    # # Show the plot
    # plt.show()

if __name__ == '__main__':
    model = plot_probability_differences()
    