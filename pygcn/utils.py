import numpy as np
import scipy.sparse as sp
import torch

"""
주어진 레이블 배열(문자열 리스트)을 원-핫 벡터로 변환하는 함수
"""


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


"""
Cora 데이터세트는 노드가 머신러닝 논문을 나타내고 간선이 논문 쌍 간의 인용을 나타내는 인용 그래프
cora.cites: 논문 인용관계
- 논문ID1 논문ID2 쌍들이 줄 단위로 기록되어, 논문 간의 인용 관계를 나타냅니다.
cora.content: 논문 내 단어 포함 여부
- 각 줄에 {논문ID, word1, word2, ..., wordN, Label} 형식으로 정보가 들어있습니다.
"""


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print("Loading {} dataset...".format(dataset))

    # 1. cora.content 파일을 읽어서 특징 행렬과 레이블을 생성
    idx_features_labels = np.genfromtxt(
        "{}{}.content".format(path, dataset), dtype=np.dtype(str)
    )
    # 원본 코드에서는 idx_features_labels를 바로 sp.csr_matrix에 넘겨서 csr 형태로 만드는데, 원소가 str으로 인식되어 정상적인 csr 형태로 저장되지 않음.
    # 따라서 str[][] 형태의 feature matrix를 float[][] 형태로 변환
    feature_sparse_matrix = [
        list(float(w) for w in d) for d in idx_features_labels[:, 1:-1]
    ]
    features = sp.csr_matrix(  # feature matrix with csr format
        feature_sparse_matrix, dtype=np.float32
    )  # feature_matrix는 sparse하다. 따라서 csr(compressed sparse row) 방식으로 다루어서 효율적으로 연산토록 함
    labels = encode_onehot(  # label이 one hot 벡터로 표현된 벡터의 리스트
        idx_features_labels[:, -1]
    )

    # 2. 노드(논문) 인덱스 매핑
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {
        j: i for i, j in enumerate(idx)
    }  # {논문ID: feature, labels 에서의 인덱스}

    # 3. cora.cites 파일을 읽어서 인접 행렬 생성
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(  # 논문 ID가 아닌 feature, labels 에서의 인덱스로 인용관계가 표현된 행렬
        list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
    ).reshape(
        edges_unordered.shape
    )
    adj = sp.coo_matrix(  # adjacency matrix(단방향만 반영) with coo format
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),  # data, (row,col)
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )

    # build symmetric adjacency matrix
    adj = (
        adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    )  # 인접행렬 대칭으로 만들어주기

    # 4. 특징 행렬(features)과 인접 행렬(adj)에 대한 정규화
    features = normalize(features)  #  각 노드마다 row-wise로 정규화
    adj = normalize(
        adj + sp.eye(adj.shape[0])
    )  # 인접행렬에 self-loop 추가 후 row-wise로 정규화

    # 5. 학습(train), 검증(val), 테스트(test)용 노드 인덱스 설정
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    # 6. NumPy 배열을 PyTorch 텐서 형태로 변환
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(
        np.where(labels)[1]
    )  # one-hot 인코딩을 정수 인코딩으로 변경
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    # 인덱스들도 PyTorch 텐서 형태로 변환
    ## PyTorch의 autograd(자동 미분) 기능과 GPU 가속을 활용하려면 모든 데이터가 torch.Tensor 형태여야 함.
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(
        labels
    )  # output의 각 행(node) 중 가장 큰 값의 인덱스
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)  # 전체 노드 중 실제 label과 예측이 일치한 비율


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
