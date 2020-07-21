from loaddata import *
from transformer_enc import *
from config import *
def train_model(pairs, model, criterion, optimizer, device, epoch):
    model.train()
    for i in range(epoch):
        for j,batch in enumerate(pairs):
            target = torch.LongTensor(batch[0]).to(device)
            x = torch.LongTensor(batch[1]).transpose(0, 1).to(device)
            out = model.forward(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            print('第%d次迭代' % i, '第%d个batch' % j, 'loss为：', loss)
def predict_model(tuple, model, device):
    model.eval()
    right = 0.
    for i in range(len(tuple[0])):
        x = torch.LongTensor(tuple[1][i]).unsqueeze(0).transpose(0,1).to(device)
        out = model.forward(x)
        pre = torch.argmax(out)
        if pre.item()==tuple[0][i]:
            right+=1
    precision = right/len(tuple[0])
    print('测试集上的精度为：', precision)


if __name__ == '__main__':
    args = TextCNNConfig()
    max_len, vocab_size, label_size, trainpairs, testtuple = load_data(args.trainfile, args.testfile, args.batch_size)
    device = torch.device(args.device)
    model = S2sTransformer(vocab_size, label_size, max_len, LearnedPositionEncoding, args.d_model, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr)

    train_model(trainpairs, model, criterion, optimizer, device, args.epoch)
    predict_model(testtuple, model, device)