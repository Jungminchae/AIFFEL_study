# Dacon MNIST competition 
## 2조 코드 공유
- 실험을 좀 편하게 할 수 있도록 만들어 봤습니다. ***run.py*** 에 들어가서 **create_cnn_model** 만 바꾸면 바로 모델 학습을 해볼 수 있습니다.  
git clone 하셔서 사용하면 됩니다.
```shell
git clone https://github.com/Jungminchae/AIFFEL_study.git
```
- 그 후 clone 하신 디렉토리 안에 들어가서 몇 개의 디렉토리만 더 만들면 됩니다.
```shell
cd AIFFEL_study/MNIST
mkdir data
mkdir data/aug
mkdir submissions
```

# Jupyter notebook에서 사용법
- keras ImageGenerator를 사용해서 augmentation을 할 수 있습니다.
- local에 저장할 수 있도록 만들었습니다.
```python
from dataloader import Dataloader
a = Dataloader()
# data 디렉토리 안 aug 디렉토리에 저장이 됩니다
a.image_generator_local_save()
```
## Augmentation 데이터 미사용
- cell 안에서 실행하면 됩니다.
- EX)
```python
!python run.py --augment_data false \
               --batch_size 32 \
               --model_save true \
               --submission_name submission_01.csv
```
## Augmentation 데이터 사용
```python
!python run.py --augment_data true \
               --batch_size 32 \
               --model_save true \
               --submission_name submission_01.csv
```

- augment_data : augmentation한 데이터를 사용할 것인지 여부
- batch_szie : batch 크기
- model_save : 학습 후 모델을 저장할 것인지 여부 , model 디렉토리에 저장됨
- submission_name : submission 파일 이름, submissions 디렉토리에 저장


