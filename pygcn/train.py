from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="Disables CUDA training."
)
parser.add_argument(
    "--fastmode",
    action="store_true",
    default=False,
    help="Validate during training pass.",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--epochs", type=int, default=200, help="Number of epochs to train."
)
parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate.")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="Weight decay (L2 loss on parameters).",
)
parser.add_argument("--hidden", type=int, default=16, help="Number of hidden units.")
parser.add_argument(
    "--dropout", type=float, default=0.5, help="Dropout rate (1 - keep probability)."
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = (
    load_data()
)  # NOTE: load_data에서 features 불러오는 코드 수정함. 수정 전과 차이있는지 실행해서 확인 필요

# Model and optimizer
model = GCN(
    nfeat=features.shape[1],  # feature 개수
    nhid=args.hidden,  # hidden layer 개수
    nclass=labels.max().item() + 1,  # 클래스 개수
    dropout=args.dropout,  # dropout rate
)
optimizer = optim.Adam(  # 경사하강법 할 때 어떤식으로 하강할지
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
)

if args.cuda:  # GPU 메모리로 이동
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()  # 학습 상태로 세팅
    optimizer.zero_grad()  # 이전에 계산되어 저장된 모든 파라미터의 기울기(gradient)를 0으로 초기화
    output = model(
        features, adj
    )  # forward 호출 : 모델에 입력 데이터를 주어 예측값(출력)을 얻음
    loss_train = F.nll_loss(
        output[idx_train], labels[idx_train]
    )  # 학습 데이터 구간의 예측 데이터와 실제 데이터간 loss 구함
    acc_train = accuracy(
        output[idx_train], labels[idx_train]
    )  # 학습 데이터 구간에서 전체 노드 중 실제 label과 예측이 일치한 비율
    loss_train.backward()  # 손실을 각 파라미터에 대한 기울기로 역전파(backpropagation) (gradient 계산)
    optimizer.step()  # 계산된 gradient를 바탕으로 모델 파라미터 업데이트

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()  # evaluation 모드로 변환
        output = model(features, adj)  # dropout 하지 않고 forward

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])  # 검증 데이터로 loss 계산
    acc_val = accuracy(output[idx_val], labels[idx_val])  # 검증 데이터로 accuracy 계산
    print(
        "Epoch: {:04d}".format(epoch + 1),
        "loss_train: {:.4f}".format(loss_train.item()),
        "acc_train: {:.4f}".format(acc_train.item()),
        "loss_val: {:.4f}".format(loss_val.item()),
        "acc_val: {:.4f}".format(acc_val.item()),
        "time: {:.4f}s".format(time.time() - t),
    )


def test():
    model.eval()  # evaluation 모드로 변환
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print(
        "Test set results:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test.item()),
    )


# Train model
t_total = time.time()
for epoch in range(
    args.epochs
):  # 한 epoch: 학습 데이터 셋에 포함된 모든 데이터들이 한 번씩 모델을 통과함
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
