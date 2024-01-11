import pandas as pd

def read_dataset(file_path):
    dataset = pd.read_csv(file_path)
    return dataset
def main():
    filename = "white_dwarf_data.csv"
    dataset = read_dataset(filename)

if __name__ == '__main__':
    main()

