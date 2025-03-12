import model
import loss
import torch
import matplotlib.pyplot as plt
import torch.optim as optim

firstSubfix = "ChosenPrefixCollision.pdf"
subfix = b""
with open(firstSubfix, 'rb') as f:
    subfix = f.read(800)  

prefix1Data = "random_1.pdf"
prefix1 = b""
with open(prefix1Data, 'rb') as f:
    prefix1 = f.read(800)  

prefix2Data = "random_2.pdf"
prefix2 = b""
with open(prefix2Data, 'rb') as f:
    prefix2 = f.read(800)

data1 = [float(x) for x in list(prefix1 + subfix)]
data2 = [float(x) for x in list(prefix2 + subfix)]
data1 = torch.tensor(data1)
data2 = torch.tensor(data2)
modelAI1 = model.ModelAI(in_channels=1, dataSize=40, layers=128)
modelAI2 = model.ModelAI(in_channels=1, dataSize=40, layers=128)

loss_fn = loss.LossFunction()

out1 = modelAI1(data1.reshape(40, 40).unsqueeze(0).unsqueeze(0))
out2 = modelAI2(data2.reshape(40, 40).unsqueeze(0).unsqueeze(0))

optimizer1 = optim.Adam(modelAI1.parameters(), lr = 0.01, weight_decay=0)
optimizer2 = optim.Adam(modelAI2.parameters(), lr = 0.01, weight_decay=0)

x = []
y = []

for i in range(100):
    # luot 1, modelAI1
    loss_1 = loss_fn(out1, out2)
    optimizer1.zero_grad()
    loss_1.backward(retain_graph=True)
    optimizer1.step()
    
    out1 = modelAI1(data1.reshape(40, 40).unsqueeze(0).unsqueeze(0))
    if out1.tolist() == out2.tolist():
        print("Phat hien va cham")
    else:
        loss_2 = loss_fn(out2, out1)
        optimizer2.zero_grad()
        loss_2.backward(retain_graph=True)
        optimizer2.step()
        out2 = modelAI2(data2.reshape(40, 40).unsqueeze(0).unsqueeze(0))
        if out1.tolist() == out2.tolist():
            print("Phat hien va cham")
    print(f"Độ sai lệch giữa 2 đầu ra kết quả: {torch.sum(torch.abs(out1 - out2))}")
    x.append(i)
    y.append(torch.sum(torch.abs(out1 - out2)).item())

out1 = modelAI1(data1.reshape(40, 40).unsqueeze(0).unsqueeze(0))
out2 = modelAI2(data2.reshape(40, 40).unsqueeze(0).unsqueeze(0))

# Vẽ đồ thị
plt.plot(x, y, marker='o', linestyle='-', color='b', label='Dữ liệu')

# Thêm tiêu đề và nhãn trục
plt.title("Đồ thị đường")
plt.xlabel("Trục X")
plt.ylabel("Trục Y")

# Hiển thị chú thích
plt.legend()

# Hiển thị đồ thị
plt.show()