from prettytable import PrettyTable
import csv

def print_results_from_csv(csv_file_path):
    """
    Prints results from a CSV file using PrettyTable, matching the desired view.

    Args:
        csv_file_path (str): The path to the CSV file.
    """
    try:
        with open(csv_file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Get the header row

            # Create a mapping of the header to more readable column names
            column_mapping = {
                "chain": "Chain",
                "metric": "Metric",
                "max_acc": "Max Accuracy",
                "file_max_acc": "File (max acc)",
                "min_acc": "Min Accuracy",
                "file_min_acc": "File (min acc)",
                "max_auc": "Max AUC",
                "file_max_auc": "File (max auc)",
                "min_auc": "Min AUC",
                "file_min_auc": "File (min auc)",
                "max_sens": "Max Sensitivity",
                "file_max_sens": "File (max sens)",
                "min_sens": "Min Sensitivity",
                "file_min_sens": "File (min sens)",
                "max_spec": "Max Specificity",
                "file_max_spec": "File (max spec)",
                "min_spec": "Min Specificity",
                "file_min_spec": "File (min spec)"
            }

            table = PrettyTable()
            for col in header:
                table.add_column(column_mapping.get(col, col), [])  # Add columns with readable names

            for row_data in reader:
                formatted_row = []
                row_dict = dict(zip(header, row_data))
                for col in header:
                    value = row_dict.get(col)
                    try:
                        # Try to convert to float and format if it's a number
                        formatted_value = f"{float(value):.4f}"
                    except ValueError:
                        # If it's not a number (e.g., file name, chain, metric), keep it as is
                        formatted_value = value
                    formatted_row.append(formatted_value)
                table.add_row(formatted_row)
            print("\nMAKSYMALNE I MINIMALNE WYNIKI DLA KAŻDEGO ŁAŃCUCHA I METRYKI:")
            print(table)

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    csv_file = 'trees_increment_extremes.csv'  # Replace with the actual path to your CSV file
    print_results_from_csv(csv_file)