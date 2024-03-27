# 파이토치 study
# 파이토치 튜토리얼
import torch
import numpy as np

# 데이터 생성
# 데이터로부터 직접 텐서를 생성
data = [[1,2] , [3,4]]
x_data = torch.tensor(data)

# Numpy 배열로 부터 텐서를 생성
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# x_data의 속성을 유지 : 2x2 행렬에 원소들은 모두 1
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

# 위와 동일하지만 원소들은 랜덤
x_rand = torch.rand_like(x_data , dtype = torch.float)
print(f"Random Tensor : \n {x_rand} \n")

# 무작위 or 상수 값 입력하기
# 생성되는 텐서의 shape는 2,3로 유지 후 값만 랜덤 , 제로 , 1로 채우기
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor : \n {rand_tensor} \n")
print(f"Ones Tensor : \n {ones_tensor} \n")
print(f"Zeros Tensor : \n {zeros_tensor} \n")

# 텐서 연산
# 원래라면 gpu가 있는지 검사를 해야하지만 실행환경이 M1 실리콘임으로 패스
tensor = torch.ones(4,4)
print(f"First row : \n {tensor[0]} \n")
print(f"First Column : \n {tensor[:, 0]} \n")
print(f"Last Column : \n {tensor[... , -1]} \n")
tensor[: , 1] = 0 # 모든 행을 선택하고 두 번째 열의 값을 0으로 채워라는 것
print(tensor)

# 텐서 합치기
# dim은 기준 : 0이면 세로를 기준으로 쌓고 , 1이면 가로를 기준으로 쌓음
t1 = torch.cat([tensor ,tensor ,tensor ,tensor] , dim = 1)
print(t1)

# 산술 연산
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1) # rand_like는 동일한 shape과 type을 가지는 랜덤값을 생성

# y2의 연산과 동일한 연산을 수행하고 결과값은 y3에 저장
torch.matmul(tensor , tensor.T , out = y3)

# 모든 원소를 sum
agg = tensor.sum()
agg_item = agg.item() # item은 tensor의 값을 스칼라 값으로 변환하는 데 사용
print(agg_item , type(agg_item))

# 텐서의 값을 변경
print(f'{tensor} \n')
tensor.add_(5) # 텐서의 모든값에 5를 +
print(tensor)

# 텐서를 Numpy 배열로 변환
t = torch.ones(5)
print(f"t : {t}")
n = t.numpy()
print(f"n : {n}")

# 반대
n = np.ones(5)
t = torch.from_numpy(n)

# Numpy 배열의 변경 사항이 텐서에 반영됨
np.add(n , 1 , out = n) # Numpy의 모든 원소에 1을 더하고 결과값을 n에 저장
print(f't : {t}')
print(f'n : {n}')