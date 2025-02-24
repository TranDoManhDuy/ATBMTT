import torch

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_link, out_size):
        self.annotations = open(data_link).readlines()
        self.annotations = [i.strip() for i in self.annotations]
        self.out_size = out_size
        
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        text, result = self.annotations[index].split()
        result = float(result)
        # text chính là kết quả của phép băm => 40 kí tự hex
        text = str(bin(int(text, 16))[2:])
        while len(text) < 169:
            text = text + "0"
        # text bây giờ là một chuỗi bit dài 169 kí tự bit 0/1, sẵn sàng đi qua model
        text_matrix = []
        for i in range(0, len(text) , 13):
            text_matrix.append([float(i) for i in text[i: i + 13]])
            
        text_matrix = torch.tensor(text_matrix).unsqueeze(0)
        # text_matrix là đầu vào model
        return text_matrix, torch.tensor([[result]])