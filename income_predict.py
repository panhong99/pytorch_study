import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

income = pd.read_csv('/Users/limpanhong/Desktop/Programing/DKU_study/open/train.csv')
print(income.head()) # 불러온 값 확인

# 데이터들의 결측치 확인
print(income.isnull().sum() , '\n')

# 순서나 크기와 같은 의미를 내포하지 않는 범주형을 다룰때는 label incoding을 하는것이 좋은데
# 그게 아닌경우는 원-핫 인코딩을 하는것이 더 좋음

# 정수형 값을 제외하고는 모두 범주형텍스트 데이터이기 때문에
# label encoding을 통해서 정수값으로 변경해줌
label_encoder = LabelEncoder()

# 여러 columns에 대해서 label Encoding 수행
# 0번째 컬럼은 id를 나타내는 것이니 제외
# 마지막은 정답 column이니 제외
# 데이터를 확인했을 때 첫번째 값이 텍스트가 아닌 경우는 반드시 숫자로 할당이 되어있는 컬럼이기 때문에
# 그런 컬럼을 제외하고 레이블 인코딩 실행
for i in income.columns[ : -1]:
    if isinstance(income[i].iloc[0] , (int , float)) or income[i].iloc[0] == 0:
        pass
    else:
        income[i] = label_encoder.fit_transform(income[i])
        print(f'{i}의 클래스 및 할당된 숫자 : {label_encoder.classes_}')
        print(f'{i}의 변환된 값 :  {income[i].unique()}')
# print(income.head()) # 인코딩이 정확하게 되었는지 확인

# target data와 feture data 분리
train_data = income.drop('Income', axis = 1) # pandas에서 뒤에 축을 기준을 잡아줘야 슬라이싱을 함
target = income['Income']

# torch로 학습을 진행시키기 위해서 tensor로 type를 변경해줌
train_data = torch.tensor(train_data.values , dtype=torch.float32)
target = torch.tensor(target.values , dtype=torch.float32)

# # 히트맵을 그려서 확인을 해봤을 때 Working_Week (Yearly)가 income과 약한 양의 상관관계에 있고
# # 나머지 변수들은 음의 상관관계에 놓여있다는 것을 확인할 수 있었음
# plt.figure(figsize = (5,5))
# sns.heatmap(income.corr() , annot = True , cmap = 'coolwarm' , fmt = '.2f' , linewidths=0.5)
# plt.title('Correlation Heatmap')
# plt.show()

# train , test split

X_train , X_test , y_train , y_test = train_test_split(train_data , target , test_size= 0.2 , random_state=42)

train_dataset = TensorDataset(X_train , y_train)
train_loader = DataLoader(train_dataset , batch_size=64 , shuffle=True)

#2024 / 03 / 26#
class MLP(nn.Module):
    def __init__(self , input_size , hidden_size , output_size):
        super(MLP , self).__init__()
        self.fc1 = nn.Linear(input_size , hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size , output_size)

    def forward(self , x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_size = len(X_train[0])
hidden_size = 64
output_size = 1

model = MLP(input_size , hidden_size , output_size)

criterion = nn.MSELoss()
optimizer  = optim.Adam(model.parameters() , lr = 0.001)


num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # 모델을 학습 모드로 설정
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()  # 기울기 초기화
        outputs = model(inputs)  # 모델에 입력 전달하여 출력 계산
        loss = criterion(outputs, targets.unsqueeze(1))  # 손실 함수 계산
        loss.backward()  # 역전파 수행
        optimizer.step()  # 최적화 수행
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

model.eval()  # 모델을 평가 모드로 설정
with torch.no_grad():
    outputs = model(X_test)
    loss = criterion(outputs, y_test.unsqueeze(1))
    print(f'Test Loss: {loss.item():.4f}')