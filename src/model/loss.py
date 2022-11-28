import numpy as np
from pathlib import Path

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
import mindspore.numpy as msnp
from mindspore import Tensor, context
from mindspore.common import dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as op
from mindspore.nn.loss.loss import LossBase


from src.model.model import PSDNet
from src.data.dataset import ms_map, dataloader
from src.utils.tools import ConfigS3DIS as cfg

class WeightCEloss(LossBase):
    """weight ce loss"""

    def __init__(self, weights, num_classes):
        super(WeightCEloss, self).__init__()
        self.weights = weights
        self.num_classes = num_classes
        self.onehot = nn.OneHot(depth=num_classes)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=False)

    def construct(self, logits, labels):
        labels = P.Concat(0)([labels, labels])
        one_hot_label = self.onehot(labels)  # [2b*n, 13]
        weights = self.weights * one_hot_label  # [2b*n, 13]
        weights = P.ReduceSum(keep_dims=False)(weights, 1)  # [2b*n]
        logit = P.cast(logits, mstype.float32)
        one_hot_label = P.cast(one_hot_label, mstype.float32)
        unweighted_loss = self.ce(logit, one_hot_label)  # [2b*n]
        weighted_loss = unweighted_loss * weights  # [2b*n]
        CE_loss = P.ReduceMean(keep_dims=False)(weighted_loss) # [1]

        return CE_loss


class JSLoss(LossBase):
    """Jensen-Shannon divergence"""

    def __init__(self, b):
        super(JSLoss, self).__init__()
        self.b = b
        self.softmax = nn.Softmax(axis=-1)
        self.norm = nn.Norm(axis=-1, keep_dims=False)

    def construct(self, logits):
        logits = logits.swapaxes(-2, -1)
        # logits = self.softmax(logits)
        logits_clean = logits[:self.b, :, :].reshape(-1,logits.shape[-1])
        logits_noise = logits[self.b:, :, :].reshape(-1,logits.shape[-1])
        p1 = P.cast(logits_clean, mstype.float32)
        p2 = P.cast(logits_noise, mstype.float32)
        # q = 1/2*(p1+p2)
        # loss_kl = p1 * P.Log()(p1/(q+1e-4)+1e-4) + p2 * P.Log()(p2/(q+1e-4)+1e-4)
        loss_cos = (1-P.ReduceSum()(p1*p2,-1)/(self.norm(p1)*self.norm(p2)))*10

        return P.ReduceMean(keep_dims=False)(loss_cos)
        

class CRLoss(LossBase):
    """CR loss"""

    def __init__(self, num_classes):
        super(CRLoss, self).__init__()
        self.onehot = nn.OneHot(depth = num_classes)
        self.relu = nn.ReLU()

    def construct(self, rs1, rs2, labels):
        label_pool_one_hot = self.onehot(labels) 
        Afinite_hot = P.matmul(label_pool_one_hot, P.Transpose()(label_pool_one_hot, (1, 0)))

        rs_map_soft = P.matmul(rs1, P.Transpose()(rs2, (1, 0))) 
        rs_map_soft = self.relu(rs_map_soft)
        rs_map_soft = P.clip_by_value(rs_map_soft, 1e-4, 1-(1e-4))
        Afinite = Afinite_hot.reshape([-1, 1])
        rs_map = rs_map_soft.reshape([-1, 1])
        loss_cr = -1.0 * P.ReduceMean()(Afinite * P.Log()(rs_map) + (1 - Afinite) * P.Log()(1 - rs_map))
        A_R = P.ReduceSum()(Afinite_hot * rs_map_soft, 1)
        loss_tjp = -1.0 * P.ReduceMean()(P.Log()(P.Div()(A_R, P.ReduceSum()(rs_map_soft, 1))))
        loss_tjr = -1.0 * P.ReduceMean()(P.Log()(P.Div()(A_R, P.ReduceSum()(Afinite_hot, 1))))
        A_R_1 = P.ReduceSum()((1-Afinite_hot) * (1-rs_map_soft), 1)

        return loss_cr + loss_tjp + loss_tjr



class PSDWithLoss(nn.Cell):
    """PSD-net with loss"""

    def __init__(self, network, weights, num_classes, ignored_label_inds, is_training):
        super(PSDWithLoss, self).__init__()
        self.network = network
        self.num_classes = num_classes
        self.ignored_label_inds = ignored_label_inds
        self.is_training = is_training
        self.b = cfg.batch_size
        self.ce_loss = WeightCEloss(weights, num_classes)
        self.kl_loss = JSLoss(cfg.batch_size)
        self.cr_loss = CRLoss(num_classes)

    def construct(self, feature, aug_feature, labels, valid_idx, input_inds, cloud_inds, p0, p1, p2, p3, p4, n0, n1, n2, n3, n4, pl0, pl1, pl2,
                  pl3, pl4, u0, u1, u2, u3, u4):
        #handle input
        xyz = [p0, p1, p2, p3, p4]
        neighbor_idx = [n0, n1, n2, n3, n4]
        sub_idx = [pl0, pl1, pl2, pl3, pl4]
        interp_idx = [u0, u1, u2, u3, u4]

        #forward
        logits, rs1, rs2 = self.network(xyz, feature, aug_feature, neighbor_idx, sub_idx, interp_idx)
        logits_clean = logits[:self.b, :, :]
        logits_noise = logits[self.b:, :, :]
        logits_clean = logits_clean.swapaxes(-2, -1).reshape((-1, self.num_classes))  # [b*n, 13]
        logits_noise = logits_noise.swapaxes(-2, -1).reshape((-1, self.num_classes))  # [b*n, 13]
        tmp = rs1.shape[-1]
        rs1 = rs1.reshape((-1, tmp))
        rs2 = rs2.reshape((-1, tmp))
         
        global_labels = P.Concat(0)([labels, labels])
        global_labels = global_labels.reshape((-1,))  # [2b, n] --> [2b*n]
        labels = labels.reshape((-1,))  # [b, n] --> [b*n]
        
        #select valid logits and labels
        valid_idx = valid_idx[0]
        global_valid_idx = P.Concat(0)([valid_idx, valid_idx])

        valid_logits_clean = P.Gather()(logits_clean, valid_idx, 0)
        valid_logits_noise = P.Gather()(logits_noise, valid_idx, 0)
        valid_logits = P.Concat(0)([valid_logits_clean, valid_logits_noise])
        rs1 = P.Gather()(rs1, global_valid_idx, 0)
        rs2 = P.Gather()(rs2, global_valid_idx, 0)

        valid_labels = P.Gather()(labels, valid_idx, 0)
        global_valid_labels = P.Gather()(global_labels, global_valid_idx, 0)

        #compute loss
        ce_loss = self.ce_loss(valid_logits, valid_labels)
        kl_loss = self.kl_loss(logits)
        cr_loss = self.cr_loss(rs1, rs2, global_valid_labels)

        loss = ce_loss + kl_loss + cr_loss

        return loss


class TrainingWrapper(nn.Cell):
    """Training wrapper."""

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def construct(self, *args):
        """Build a forward graph"""
        weights = self.weights
        loss, logits = self.network(*args)
        sens = op.Fill()(op.DType()(loss), op.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        return F.depend(loss, self.optimizer(grads)), logits


def get_param_groups(network):
    """Param groups for optimizer."""
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith('.bias'):
            # all bias not using weight decay
            no_decay_params.append(x)
        elif parameter_name.endswith('.gamma'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        elif parameter_name.endswith('.beta'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        else:
            decay_params.append(x)

    return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]

