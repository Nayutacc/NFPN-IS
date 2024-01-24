from model.ctNet.ctNet import Nest_U_Net

Network = Nest_U_Net

if __name__ == '__main__':
    import torch
    x = torch.ones((1, 1, 512, 512)).to('cuda')
    model = Network().to('cuda')
    model.eval()
    o = model(x)
    print(model)
