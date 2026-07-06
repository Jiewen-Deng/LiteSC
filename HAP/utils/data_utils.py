"""
@Author: JW Deng
@Time: 2026/7/6 17:00
"""
import torch
from dataset import EurDataset, collate_data

def get_transformer_dataloader(dataset, train_batch_size, test_batch_size):
    train_eur = EurDataset('train')
    trainloader = torch.utils.data.DataLoader(train_eur, batch_size=train_batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    test_eur = EurDataset('test')
    testloader = torch.utils.data.DataLoader(test_eur, batch_size=test_batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    return trainloader, testloader

def get_hessianloader(dataset, hessian_batch_size):
    if dataset == 'text':
        hessian_dataset = EurDataset('train')
        hessian_loader = torch.utils.data.DataLoader(
            hessian_dataset,
            batch_size=hessian_batch_size,
            shuffle=False,  # Hessian calculation usually doesn't need shuffle
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_data
        )
    else:
            raise ValueError("No valid dataset is given.")
    return hessian_loader

def init_dataloader(config):
    trainloader, testloader = get_transformer_dataloader(dataset=config.dataset,
                                                             train_batch_size=config.batch_size_train,
                                                             test_batch_size=config.batch_size_test)
    return trainloader, testloader