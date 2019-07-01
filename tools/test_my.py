import torch
from mmdet.models.losses import  RepMetLoss

if __name__ == "__main__":
    print("Simple test of emb loss")

    repmet = RepMetLoss(N=3, k=2, emb_size=2)


    l = [1, 0, 0, 1, 2]
    d = [[[1, 0.00], [1, 0.00], [1, 0.00]], #C0K0, C0K1, C1K0 ... C2K1
         [[0.001, 0.001], [1, 1], [1, 1]],
         [[0.001, 0.001], [1, 1], [1, 1]],
         [[0.001, 0.002], [1, 1], [1, 1]],
         [[.6, 1], [.6, 1], [.5, 0.001]]]

    d = [[1, 0.00], [1, 0.00], [1, 0.00],[1, 1], [1, 1]]

    d = torch.autograd.Variable(torch.Tensor(d).cuda(), requires_grad=True)
    l = torch.autograd.Variable(torch.Tensor(l).cuda(), requires_grad=False)

    loss = repmet(d, l)
    print('loss:',loss)

    pred = repmet.inference(d)
    print('pred:',pred)

    reps = repmet.get_reps()
    print('reps:',reps)
    print('done')