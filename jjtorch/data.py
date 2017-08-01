from torch.utils.data.dataset import Dataset


class MultiTensorDataset(Dataset):
    def __init__(self, data_tensor_list, target_tensor):
        for data_tensor in data_tensor_list:
            assert data_tensor.size(0) == target_tensor.size(0)

        self.data_tensor_list = data_tensor_list
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        feats = [data_tensor[index] for data_tensor in self.data_tensor_list]
        return feats+[self.target_tensor[index]]

    def __len__(self):
        return self.target_tensor.size(0)
