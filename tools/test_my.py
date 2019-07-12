import torch
from mmdet.models.losses import  RepMetLoss
from mmdet.models.bbox_heads import SharedFCRepMetBBoxHead

from mmdet.models.builder import  build_head


def test1():
    print("Simple test of emb loss")

    repmet = RepMetLoss(N=3, k=2, emb_size=2)
    repmet.init_reps()


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

# import hiddenlayer as hl
def test2():
    print("Simple test of emb loss")

    # type='SharedFCRepMetBBoxHead',
    bbox_head_dic=dict(type='SharedFCRepMetBBoxHead')

    # bbox_head = build_head(bbox_head_dic)
    import pdb;pdb.set_trace()
    # pdb.set_trace()
    bbox_head = SharedFCRepMetBBoxHead().to('cuda:0')
    bbox_head.init_weights()

    sample_feature_rois = torch.rand(4, 256,7,7)
    print('sample_feature_rois:',sample_feature_rois[:,8,:,:])

    input = torch.autograd.Variable(sample_feature_rois.cuda(), requires_grad=True)
    cls_score, bbox_pred = bbox_head.forward(input)
    #print('cls_score:',cls_score)
    #print('bbox_pred shape :', bbox_pred.shape)
    #print('bbox_pred:', bbox_pred)

    label_weights = torch.rand(4).cuda()
    bbox_targets = torch.rand(4, 4).cuda()
    bbox_weights = torch.rand(4, 4).cuda()
    l = [5, 0, 1, 80]
    labels = torch.autograd.Variable(torch.Tensor(l).cuda().long(), requires_grad=False)

    loss = bbox_head.loss(input,bbox_pred,labels,label_weights,bbox_targets,bbox_weights,)
    print('loss:', loss)
    """
    graph = hl.build_graph(bbox_head, input)
    graph = graph.build_dot()
    graph.render("/opt/zjw/my_project/mmdetection/bbox_head.png", view=False, format='png')
    """

if __name__ == "__main__":
    test1()
    test2()