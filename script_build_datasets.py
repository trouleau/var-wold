import pickle
import os
from tsvar.preprocessing import Dataset


def make_dataset(in_path, top, out_path):
    print()
    if os.path.exists(in_path):
        print(f'Make dataset: {in_path}...')
        dataset = Dataset(in_path, top)
        print((f'Dataset has {dataset.dim:d} dimensions '
               f'and {sum(map(len, dataset.timestamps)):,d} events'))
        with open(out_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f'Saved as: {out_path}')
    else:
        print(f"Warning: Dataset '{in_path}' not found!")


if __name__ == "__main__":

    # make_dataset(
    #     in_path=os.path.join('data', 'email-Eu-core-temporal.txt.gz'),
    #     top=100,
    #     out_path=os.path.join('data', 'email-Eu-core-temporal-top100.pk')
    # )

    # make_dataset(
    #     in_path=os.path.join('data', 'soc-sign-bitcoinalpha.csv.gz'),
    #     top=100,
    #     out_path=os.path.join('data', 'soc-sign-bitcoinalpha-top100.pk')
    # )

    # make_dataset(
    #     in_path=os.path.join('data', 'soc-sign-bitcoinotc.csv.gz'),
    #     top=100,
    #     out_path=os.path.join('data', 'soc-sign-bitcoinotc-top100.pk')
    # )

    # make_dataset(
    #     in_path=os.path.join('data/enron/', 'enron_dataset_splitted_receivers.csv.gz'),
    #     top=100,
    #     out_path=os.path.join('data/enron/', 'enron-top100.pk')
    # )

    make_dataset(
        in_path=os.path.join('data/', 'wiki-talk-temporal.txt.gz'),
        top=100,
        out_path=os.path.join('data/', 'wiki-talk-temporal.pk')
    )
