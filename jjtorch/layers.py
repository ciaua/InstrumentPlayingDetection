import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch.legacy.nn import SpatialCrossMapLRN as SpatialCrossMapLRNOld
from torch.nn import Module


class GaussianWrong(nn.Module):
    def __init__(self):
        """

        Output
        ------
        For each point (b, l, f), the value is the weighted sum of
        x[b, l, :num_frames] and g
        where g has shape=(num_frames, ) and it forms a Gaussian function
        centered at f with std=sigma[b, l, f]

        output shape=(batchsize, num_labels, num_frames)
        """
        # shape=(num_frames, num_frames)
        super(GaussianWrong, self).__init__()
        self.g_base_dict = dict()

    def compute_g_base(self, num_frames):
        """
        Example:
            If num_frames=3,
            [[0, 1, 2],
             [-1, 0, 1],
             [-2, -1, 0]]
        """
        g_base = torch.zeros(num_frames, num_frames)
        # print(g_base.size())
        tt = torch.arange(0, num_frames)
        # print(tt.size())
        for ii in range(num_frames):
            g_base[ii] = tt
            tt = tt-1

        return g_base

    def make_gaussian_mask(self, sigma):
        """
        sigma: torch.FloatTensor
            shape=(batchsize, num_labels, num_frames)

        gaussian_mask:
            shape=(batchsize, num_labels, num_frames, num_frames)

            gaussian_mask[b, l, f, :] is the Gaussian filter formed
            by sigma[b, l, f] centered at the frame f
        """
        s0, s1, s2 = sigma.size()
        # g = torch.zeros((s0, s1, s2, s2))

        try:
            g_base = self.g_base_dict[s2]
        except Exception:
            g_base = self.compute_g_base(s2)
            self.g_base_dict[s2] = g_base

        # shape=(bs, num_labels, num_frames, num_frames)
        bone = Variable(g_base.view(1, 1, s2, s2).expand(s0, s1, s2, s2).cuda())
        # bone = Variable(g_base.view(1, 1, s2, s2).expand(s0, s1, s2, s2))

        # shape=(bs, num_labels, num_frames, num_frames)
        sigma_ex = sigma.view(s0, s1, s2, 1).expand_as(bone)

        s = sigma_ex*(2.*np.pi)**0.5
        s = s.reciprocal()
        # ones = Variable(torch.ones(s_.size()))

        # ones = Variable(torch.ones(s_.size()).cuda())

        # s = ones/s_

        mask = s*torch.exp(-bone**2/(2.*(sigma_ex**2)))

        return mask

    def forward(self, x, sigma):
        """
        x:
            shape=(batchsize, num_labels, num_frames)

        sigma: torch.FloatTensor
            shape=(batchsize, num_labels, num_frames)
        """
        s0, s1, s2 = x.size()

        # shape=(bs, num_labels, num_frames, num_frames)
        gaussian_mask = self.make_gaussian_mask(sigma)

        # shape=(bs, num_labels, num_frames, num_frames)
        x_ex = x.view(s0, s1, s2, 1).expand_as(gaussian_mask)
        # ### This is wrong but strangely it performs better

        # shape=(bs, num_labels, num_frames)
        weighted = (x_ex*gaussian_mask).sum(dim=3).view(s0, s1, s2)
        # print(gaussian_mask.data.cpu().numpy()[0, 0, 0, :])
        # print(x_ex.data.cpu().numpy()[0, 0, 0, :])
        # print(weighted.data.cpu().numpy()[0, 0, :])
        # raw_input(123)

        return weighted


class GaussianX(nn.Module):
    def __init__(self):
        """

        Output
        ------
        For each point (b, l, f), the value is the weighted sum of
        x[b, l, :num_frames] and g
        where g has shape=(num_frames, ) and it forms a Gaussian function
        centered at f with std=sigma[b, l, f]

        output shape=(batchsize, num_labels, num_frames)
        """
        # shape=(num_frames, num_frames)
        super(GaussianX, self).__init__()
        self.g_base_dict = dict()

    def compute_g_base(self, num_frames):
        """
        Example:
            If num_frames=3,
            [[0, 1, 2],
             [-1, 0, 1],
             [-2, -1, 0]]
        """
        g_base = torch.zeros(num_frames, num_frames)
        # print(g_base.size())
        tt = torch.arange(0, num_frames)
        # print(tt.size())
        for ii in range(num_frames):
            g_base[ii] = tt
            tt = tt-1

        return g_base

    def make_gaussian_mask(self, sigma):
        """
        sigma: torch.FloatTensor
            shape=(batchsize, num_labels, num_frames)

        gaussian_mask:
            shape=(batchsize, num_labels, num_frames, num_frames)

            gaussian_mask[b, l, f, :] is the Gaussian filter formed
            by sigma[b, l, f] centered at the frame f
        """
        s0, s1, s2 = sigma.size()
        # g = torch.zeros((s0, s1, s2, s2))

        try:
            g_base = self.g_base_dict[s2]
        except Exception:
            g_base = self.compute_g_base(s2)
            self.g_base_dict[s2] = g_base

        # shape=(bs, num_labels, num_frames, num_frames)
        bone = Variable(g_base.view(1, 1, s2, s2).expand(s0, s1, s2, s2).cuda())
        # bone = Variable(g_base.view(1, 1, s2, s2).expand(s0, s1, s2, s2))

        # shape=(bs, num_labels, num_frames, num_frames)
        sigma_ex = sigma.view(s0, s1, s2, 1).expand_as(bone)

        s = sigma_ex*(2.*np.pi)**0.5
        s = s.reciprocal()
        # ones = Variable(torch.ones(s_.size()))

        # ones = Variable(torch.ones(s_.size()).cuda())

        # s = ones/s_

        mask = s*torch.exp(-bone**2/(2.*(sigma_ex**2)))

        return mask

    def forward(self, x, sigma):
        """
        x:
            shape=(batchsize, num_labels, num_frames)

        sigma: torch.FloatTensor
            shape=(batchsize, num_labels, num_frames)
        """
        s0, s1, s2 = x.size()

        # shape=(bs, num_labels, num_frames, num_frames)
        gaussian_mask = self.make_gaussian_mask(sigma)

        # shape=(bs, num_labels, num_frames, num_frames)
        x_ex = x.view(s0, s1, 1, s2).expand_as(gaussian_mask)

        # shape=(bs, num_labels, num_frames)
        weighted = (x_ex*gaussian_mask).sum(dim=3).view(s0, s1, s2)
        # print(gaussian_mask.data.cpu().numpy()[0, 0, 0, :])
        # print(x.data.cpu().numpy()[0, 0, :])
        # print(weighted.data.cpu().numpy()[0, 0, :])
        # raw_input(123)

        return weighted


class BumpX(nn.Module):
    def __init__(self, d):
        """

        Output
        ------
        For each point (b, l, f), the value is the weighted sum of
        x[b, l, :num_frames] and g
        where g has shape=(num_frames, ) and it forms a Gaussian function
        centered at f with std=sigma[b, l, f]

        output shape=(batchsize, num_labels, num_frames)
        """
        # shape=(num_frames, num_frames)
        super(BumpX, self).__init__()
        self.bone_base_dict = dict()
        self.d = d

    def compute_bone_base(self, num_frames):
        """
        Example:
            If num_frames=3,
            [[0, 1, 2],
             [-1, 0, 1],
             [-2, -1, 0]]
        """
        bone_base = torch.zeros(num_frames, num_frames)
        # print(g_base.size())
        tt = torch.arange(0, num_frames)
        # print(tt.size())
        for ii in range(num_frames):
            bone_base[ii] = tt
            tt = tt-1

        return bone_base

    def ff(self, t):
        temp = torch.clamp(F.softplus(t), min=1e-6)
        return torch.exp(-1.*temp.reciprocal())

    def gg(self, t):
        ft = self.ff(t)
        return ft/(ft+self.ff(1.-t))

    def hh(self, x, aa, bb):
        hx = self.gg((x-aa**2)/(bb**2-aa**2))
        return hx

    def kk(self, x, aa, bb):
        kx = self.hh(x**2, aa, bb)
        return kx

    def ll(self, x, aa, bb):
        lx = 1.-self.kk(x, aa, bb)
        return lx

    def make_bump_mask(self, aa, d):
        """
        aa: torch.FloatTensor
            shape=(batchsize, num_labels, num_frames)

        d: float
            bb = aa+d

        gaussian_mask:
            shape=(batchsize, num_labels, num_frames, num_frames)

            gaussian_mask[b, l, f, :] is the Gaussian filter formed
            by sigma[b, l, f] centered at the frame f
        """
        s0, s1, s2 = aa.size()
        # g = torch.zeros((s0, s1, s2, s2))

        try:
            bone_base = self.bone_base_dict[s2]
        except Exception:
            bone_base = self.compute_bone_base(s2)
            self.bone_base_dict[s2] = bone_base

        # shape=(bs, num_labels, num_frames, num_frames)
        bone = Variable(
            bone_base.view(1, 1, s2, s2).expand(s0, s1, s2, s2).cuda())
        # bone = Variable(g_base.view(1, 1, s2, s2).expand(s0, s1, s2, s2))

        # shape=(bs, num_labels, num_frames, num_frames)
        aa_ex = aa.view(s0, s1, s2, 1).expand_as(bone)
        bb = aa+float(d)
        bb_ex = bb.view(s0, s1, s2, 1).expand_as(bone)

        # Make bump from Loring Tu

        mask = self.ll(bone, aa_ex, bb_ex)

        return mask

    def forward(self, x, aa):
        """
        x:
            shape=(batchsize, num_labels, num_frames)

        sigma: torch.FloatTensor
            shape=(batchsize, num_labels, num_frames)
        """
        s0, s1, s2 = x.size()

        # shape=(bs, num_labels, num_frames, num_frames)
        bump_mask = self.make_bump_mask(aa, self.d)

        # shape=(bs, num_labels, num_frames, num_frames)
        x_ex = x.view(s0, s1, 1, s2).expand_as(bump_mask)

        # print(x_ex[0, 0, 0, :].data.cpu().numpy())
        # print(bump_mask[0, 0, 0, :].data.cpu().numpy())
        # raw_input(123)

        # shape=(bs, num_labels, num_frames)
        weighted_sum = (x_ex*bump_mask).sum(dim=3).view(s0, s1, s2)
        bump_mask_sum = (bump_mask).sum(dim=3).view(s0, s1, s2)

        weighted_sum_normalized = weighted_sum/bump_mask_sum

        return weighted_sum_normalized


# Partition of unity
class BumpXPOU(nn.Module):
    def __init__(self, d):
        """

        Output
        ------
        For each point (b, l, f), the value is the weighted sum of
        x[b, l, :num_frames] and g
        where g has shape=(num_frames, ) and it forms a Gaussian function
        centered at f with std=sigma[b, l, f]

        output shape=(batchsize, num_labels, num_frames)
        """
        # shape=(num_frames, num_frames)
        super(BumpXPOU, self).__init__()
        self.bone_base_dict = dict()
        self.d = d

    def compute_bone_base(self, num_frames):
        """
        Example:
            If num_frames=3,
            [[0, 1, 2],
             [-1, 0, 1],
             [-2, -1, 0]]
        """
        bone_base = torch.zeros(num_frames, num_frames)
        # print(g_base.size())
        tt = torch.arange(0, num_frames)
        # print(tt.size())
        for ii in range(num_frames):
            bone_base[ii] = tt
            tt = tt-1

        return bone_base

    def ff(self, t):
        temp = torch.clamp(F.softplus(t), min=1e-6)
        return torch.exp(-1.*temp.reciprocal())

    def gg(self, t):
        ft = self.ff(t)
        return ft/(ft+self.ff(1.-t))

    def hh(self, x, aa, bb):
        hx = self.gg((x-aa**2)/(bb**2-aa**2))
        return hx

    def kk(self, x, aa, bb):
        kx = self.hh(x**2, aa, bb)
        return kx

    def ll(self, x, aa, bb):
        lx = 1.-self.kk(x, aa, bb)
        return lx

    def make_bump_mask(self, aa, d):
        """
        aa: torch.FloatTensor
            shape=(batchsize, num_labels, num_frames)

        d: float
            bb = aa+d

        gaussian_mask:
            shape=(batchsize, num_labels, num_frames, num_frames)

            gaussian_mask[b, l, f, :] is the Gaussian filter formed
            by sigma[b, l, f] centered at the frame f
        """
        s0, s1, s2 = aa.size()
        # g = torch.zeros((s0, s1, s2, s2))

        try:
            bone_base = self.bone_base_dict[s2]
        except Exception:
            bone_base = self.compute_bone_base(s2)
            self.bone_base_dict[s2] = bone_base

        # shape=(bs, num_labels, num_frames, num_frames)
        bone = Variable(
            bone_base.view(1, 1, s2, s2).expand(s0, s1, s2, s2).cuda())
        # bone = Variable(g_base.view(1, 1, s2, s2).expand(s0, s1, s2, s2))

        # shape=(bs, num_labels, num_frames, num_frames)
        aa_ex = aa.view(s0, s1, s2, 1).expand_as(bone)
        bb = aa+float(d)
        bb_ex = bb.view(s0, s1, s2, 1).expand_as(bone)

        # Make bump from Loring Tu

        mask = self.ll(bone, aa_ex, bb_ex)

        return mask

    def forward(self, x, aa):
        """
        x:
            shape=(batchsize, num_labels, num_frames)

        sigma: torch.FloatTensor
            shape=(batchsize, num_labels, num_frames)
        """
        s0, s1, s2 = x.size()

        # shape=(bs, num_labels, num_frames, num_frames)
        bump_mask = self.make_bump_mask(aa, self.d)

        # shape=(bs, num_labels, num_frames, num_frames)
        x_ex = x.view(s0, s1, s2, 1).expand_as(bump_mask)

        # print(x_ex[0, 0, 0, :].data.cpu().numpy())
        # print(bump_mask[0, 0, 0, :].data.cpu().numpy())
        # raw_input(123)

        # shape=(bs, num_labels, num_frames)
        weighted_sum = (x_ex*bump_mask).sum(dim=2).view(s0, s1, s2)
        bump_mask_sum = (bump_mask).sum(dim=2).view(s0, s1, s2)

        weighted_sum_pou = weighted_sum/bump_mask_sum

        return weighted_sum_pou


class GaussianXPOU(nn.Module):
    def __init__(self):
        """

        Output
        ------
        For each point (b, l, f), the value is the weighted sum of
        x[b, l, :num_frames] and g
        where g has shape=(num_frames, ) and it forms a Gaussian function
        centered at f with std=sigma[b, l, f]

        output shape=(batchsize, num_labels, num_frames)
        """
        # shape=(num_frames, num_frames)
        super(GaussianXPOU, self).__init__()
        self.g_base_dict = dict()

    def compute_g_base(self, num_frames):
        """
        Example:
            If num_frames=3,
            [[0, 1, 2],
             [-1, 0, 1],
             [-2, -1, 0]]
        """
        g_base = torch.zeros(num_frames, num_frames)
        # print(g_base.size())
        tt = torch.arange(0, num_frames)
        # print(tt.size())
        for ii in range(num_frames):
            g_base[ii] = tt
            tt = tt-1

        return g_base

    def make_gaussian_mask(self, sigma):
        """
        sigma: torch.FloatTensor
            shape=(batchsize, num_labels, num_frames)

        gaussian_mask:
            shape=(batchsize, num_labels, num_frames, num_frames)

            gaussian_mask[b, l, f, :] is the Gaussian filter formed
            by sigma[b, l, f] centered at the frame f
        """
        s0, s1, s2 = sigma.size()
        # g = torch.zeros((s0, s1, s2, s2))

        try:
            g_base = self.g_base_dict[s2]
        except Exception:
            g_base = self.compute_g_base(s2)
            self.g_base_dict[s2] = g_base

        # shape=(bs, num_labels, num_frames, num_frames)
        bone = Variable(g_base.view(1, 1, s2, s2).expand(s0, s1, s2, s2).cuda())
        # bone = Variable(g_base.view(1, 1, s2, s2).expand(s0, s1, s2, s2))

        # shape=(bs, num_labels, num_frames, num_frames)
        sigma_ex = sigma.view(s0, s1, s2, 1).expand_as(bone)

        s = sigma_ex*(2.*np.pi)**0.5
        s = s.reciprocal()
        # ones = Variable(torch.ones(s_.size()))

        # ones = Variable(torch.ones(s_.size()).cuda())

        # s = ones/s_

        mask = s*torch.exp(-bone**2/(2.*(sigma_ex**2)))

        return mask

    def forward(self, x, sigma):
        """
        x:
            shape=(batchsize, num_labels, num_frames)

        sigma: torch.FloatTensor
            shape=(batchsize, num_labels, num_frames)
        """
        s0, s1, s2 = x.size()

        # shape=(bs, num_labels, num_frames, num_frames)
        gaussian_mask = self.make_gaussian_mask(sigma)

        # shape=(bs, num_labels, num_frames, num_frames)
        x_ex = x.view(s0, s1, s2, 1).expand_as(gaussian_mask)

        # shape=(bs, num_labels, num_frames)
        weighted_sum = (x_ex*gaussian_mask).sum(dim=2).view(s0, s1, s2)
        mask_sum = gaussian_mask.sum(dim=2).view(s0, s1, s2)

        weighted_sum_pou = weighted_sum/mask_sum

        # print(gaussian_mask.data.cpu().numpy()[0, 0, 0, :])
        # print(x.data.cpu().numpy()[0, 0, :])
        # print(weighted.data.cpu().numpy()[0, 0, :])
        # raw_input(123)

        return weighted_sum_pou


def get_max_mask(arr, dim):
    # size = arr.size()
    # new_size = tuple([term for term in size if term != dim])
    M, idx = arr.max(dim=dim)
    mask = (arr == M.expand_as(arr)).float()

    normed_mask = mask/mask.sum(dim=dim).expand_as(mask)

    return normed_mask


# Local response normalization layer
# From https://github.com/pytorch/pytorch/issues/653
# function interface, internal, do not use this one

class SpatialCrossMapLRNFunc(Function):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        self.save_for_backward(input)
        self.lrn = SpatialCrossMapLRNOld(
            self.size, self.alpha, self.beta, self.k)
        self.lrn.type(input.type())
        return self.lrn.forward(input)

    def backward(self, grad_output):
        input, = self.saved_tensors
        return self.lrn.backward(input, grad_output)


# use this one instead
class SpatialCrossMapLRN(Module):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        super(SpatialCrossMapLRN, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        return SpatialCrossMapLRNFunc(
            self.size, self.alpha, self.beta, self.k)(input)
