import torch
import os

from GreedyInfoMax.audio.data import librispeech


def get_libri_dataloaders(opt):
    """
    creates and returns the Libri dataset and dataloaders,
    either with train/val split, or train+val/test split
    :param opt:
    :return: train_loader, train_dataset,
    test_loader, test_dataset - corresponds to validation or test set depending on opt.validate
    """

    num_cpus = os.cpu_count()

    num_workers = num_cpus - 1 if num_cpus > 1 else 1  # old: #1 #16

    print("num_workers: ", num_workers)

    if opt.validate:
        print("Using Train / Val Split")
        train_dataset = librispeech.LibriDataset(
            opt,
            os.path.join(
                opt.data_input_dir,
                "LibriSpeech/train-clean-100",
            ),
            os.path.join(
                opt.data_input_dir, "LibriSpeech100_labels_split/train_val_train.txt"
            ),
        )

        test_dataset = librispeech.LibriDataset(
            opt,
            os.path.join(
                opt.data_input_dir,
                "LibriSpeech/train-clean-100",
            ),
            os.path.join(
                opt.data_input_dir, "LibriSpeech100_labels_split/train_val_val.txt"
            ),
        )

    else:
        print("Using Train+Val / Test Split")
        train_dataset = librispeech.LibriDataset(
            opt,
            os.path.join(
                opt.data_input_dir,
                "LibriSpeech/train-clean-100",
            ),
            os.path.join(
                opt.data_input_dir, "LibriSpeech100_labels_split/train_split.txt"
            ),
        )

        test_dataset = librispeech.LibriDataset(
            opt,
            os.path.join(
                opt.data_input_dir,
                "LibriSpeech/train-clean-100",
            ),
            os.path.join(
                opt.data_input_dir, "LibriSpeech100_labels_split/test_split.txt"
            ),
        )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size_multiGPU,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=opt.batch_size_multiGPU,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )

    return train_loader, train_dataset, test_loader, test_dataset
