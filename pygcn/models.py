import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):  # PyTorch의 모든 신경망은 torch.nn.Module 을 상속받아 정의됨
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(
            self.gc1(x, adj)
        )  # 첫번째 layer 통과 후, 복잡한 패턴의 학습을 위한 activation func
        x = F.dropout(
            x, self.dropout, training=self.training
        )  # overfitting을 막기 위해 의도적으로 데이터에 noise를 줌
        x = self.gc2(x, adj)  # 두번째 layer 통과, [Node, nclass]
        # 벡터가 주어졌을 때, 각 원소를 0 이상 1 이하의 실수로 변환하여 결과값의 총합이 1이 되도록 한 값에 로그를 적용한 것
        # 모델의 출력을 확률 벡터로 바꿔주면서도 계산의 안정성 도모, NLLLoss를 loss함수로 함께 많이 이용
        return F.log_softmax(x, dim=1)
