import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss
from ..registry import LOSSES



@LOSSES.register_module
class RepMetLoss(nn.Module):

    def __init__(self,N, k, emb_size, alpha=1.0, sigma=0.5,
                 use_sigmoid=False,
                 lossclass_weight=0.5,
                 lossdistanc_weight=0.5,
                 loss_weight=1.0):
        super(RepMetLoss, self).__init__()
        # add by zjw
        self.N = N
        self.k = k
        self.emb_size = emb_size
        self.alpha = alpha
        self.sigma = sigma

        # TODO mod this from hardcoded with the device
        self.reps = nn.Parameter(F.normalize(torch.randn(N*k, emb_size, dtype=torch.float).cuda()))

        # assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight
        self.lossclass_weight = lossclass_weight
        self.lossdistanc_weight = lossdistanc_weight

    def inference(self,input):
        """
        :param input: x
        :return inference x ,result lables
        """
        distances = euclidean_distance(input, self.reps)
        probs = torch.exp(- distances / (2 * self.sigma ** 2))
        # Eqn 2. of repmet paper
        hard_probs, _ = probs.view(-1, self.N, self.k).max(2)
        # Eqn 3. of repmet paper
        # n_samples , N(class_num)
        back_p = 1 - hard_probs.max(1)[0]  # todo useful for detection
        # torch.unsqueeze(back_p,1)
        # n_sample * (N_classes+1)
        probs_withback = torch.cat((back_p.unsqueeze(-1),hard_probs), 1)
        _, pred = probs_withback.max(1)
        return  pred

    def forward(self, input, target):

        """
        Equation (4) of repmet paper
        :param dists: n_samples x n_classes x n_k
        :param labels: n_samples
        :return: loss1+loss2
        """
        # batch size
        self.n_samples = target.size(0)# len(target)
        # distances = euclidean_dist(input, F.normalize(self.reps))
        # --------------------distances---------------------------
        distances = euclidean_distance(input, self.reps)

        valmax, argmax = distances.max(-1)
        valmax, argmax = valmax.max(-1)
        valmax += 10 # good

        # make mask with ones where correct class, zeros otherwise
        mask = make_one_hot(target, n_classes=self.N).cuda()
        # n_class , target * k     -->  target ,N_class*k
        mask_cor = mask.transpose(0, 1).repeat(1, self.k).view(-1, self.n_samples).transpose(0, 1)
        mask_inc = ~mask_cor

        cor = distances + (valmax*mask_inc.float())
        inc = distances + (valmax*mask_cor.float())
        # min_cor,[n_smaples]
        # min_inc,[n_smaples] if [targets] != 0 indexs   min_inc[if [targets] != 0 indexs]
        min_cor, _ = cor.min(1)
        min_inc, _ = inc.min(1)

        # Eqn. 4 of repmet paper
        losses = F.relu(min_cor - min_inc + self.alpha)

        losses_la = torch.where(target == 0, target, losses)

        # mean the sample losses over the batch
        loss_distanc = self.lossdistanc_weight * torch.mean(losses_la)

        # --------------------classes labels-------------
        # Eqn. 1 of repmet paper
        probs = torch.exp(- distances / (2 * self.sigma ** 2))

        # Eqn 2. of repmet paper
        hard_probs, _ = probs.view(-1, self.N, self.k).max(2)

        # Eqn 3. of repmet paper
        # n_samples , N(class_num)
        back_p = 1 - hard_probs.max(1)[0]# todo useful for detection
        # torch.unsqueeze(back_p,1)
        # n_sample * (N_classes+1)
        probs_back = torch.cat((back_p.unsqueeze(-1),hard_probs), 1)

        # avg_factor=None can keeep means() loss
        loss_class = self.lossclass_weight * cross_entropy(probs_back, target,
                                   weight=self.lossclass_weight,
                                   reduction='mean', avg_factor=None)

        total_loss = self.loss_weight * (loss_distanc + loss_class)

        # _, pred = soft_probs.max(1)
        # acc = pred.eq(target.squeeze()).float().mean()

        return total_loss

    def get_reps(self):
        return self.reps.data.cpu().detach().numpy()

    # set reps some class or  all class
    def set_reps(self, reps, start=None, stop=None):
        if start is not None and stop is not None:
            self.reps.data[start:stop] = torch.Tensor(reps).cuda().float()
        else:
            # self.reps = nn.Parameter(torch.Tensor(reps, dtype=torch.float).cuda())
            self.reps.data = torch.Tensor(reps).cuda().float()


def euclidean_distance(x, y):
    # Compute euclidean distance between two tensors
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception("size mismatch")
    # n,d -- > n,1,d  -- > n,m,d
    x = x.unsqueeze(1).expand(n, m, d)
    # 1,m,d -- > n,m,d
    y = y.unsqueeze(0).expand(n, m, d)
    # n,m
    return torch.pow(x - y, 2).sum(2)


def make_one_hot(labels, n_classes):
    """
    :param labels: the labels in int form
    :param n_classes: the number of classes
    :return: a one hot vector with these class labels
    """
    one_hot = torch.zeros((labels.size(-1), n_classes))
    return one_hot.scatter_(1, torch.unsqueeze(labels, 1).long().cpu(), 1).byte()


def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    # element-wise losses
    loss = F.cross_entropy(pred, label.long(), reduction='none')
    # apply weights and do the reduction
    #if weight is not None:
    #    weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss
