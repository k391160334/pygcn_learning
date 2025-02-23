Graph Convolutional Networks in PyTorch
====

기존 코드에 주석을 추가하였습니다.
열심히 배우고 있는 과정에서 진행한 작업이어서 오류가 있을 수 있습니다. 이슈 발견이나 의견 공유는 언제든 메일로 남겨주세요 😊.

I added some korean comments to the original code.
Since this was done as part of my learning process, there may be some errors. If you discover any issues or would like to share feedback, please feel free to send me an email anytime. Thank you!

### Usage
```
python train.py
```

### Additional changes
1. 코드 스타일을 변경했습니다.
2. features matrix를 csr 형태로 구성하면서 0인 item 모두 저장되는 오류를 해결했습니다.


1. Updated the code style.
2. Fixed an error that caused all zero items to be saved when constructing the features matrix in CSR format.