import torch

class DocLoss(torch.nn.Module):
    def __init__(self, batchsize, classes_num):
        super(DocLoss, self).__init__()
        self.batchsize = batchsize
        self.classes_num = classes_num

    def forward(self, outputs):
        loss = 1/(self.batchsize*self.classes_num)*(self.batchsize**2)*torch.sum(((outputs-torch.mean(outputs,1,True))**2)/((self.batchsize-1)**2))
        return loss
