import torch

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,encodings,labels=[],use_labels=False):
        self.encodings = encodings
        self.labels = labels
        self.use_labels=use_labels

    def __getitem__(self, idx):
        if self.use_labels==True:
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['stance'] = torch.tensor(self.labels[idx])
        else:
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])
