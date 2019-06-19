import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm as wn
import numpy as np


def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    # Pytorch ordering
    axis = len(x.size()) - 3
    return F.elu(torch.cat([x, -x], dim=axis))


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def discretized_mix_logistic_loss(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])  # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    # log_scales = torch.max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)

    coeffs = F.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
          * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
          coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = F.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (
    log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    return -torch.sum(log_sum_exp(log_probs))


def discretized_mix_logistic_loss_1d(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 3)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2])  # 2 for mean, scale
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False)

    # means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = F.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (
    log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    return -torch.sum(log_sum_exp(log_probs))


def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda: one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return Variable(one_hot)


def sample_from_discretized_mix_logistic_1d(l, nr_mix):
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [1]  # [3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2])  # for mean, scale

    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda: temp = temp.cuda()
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    u = torch.FloatTensor(means.size())
    if l.is_cuda: u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    u = Variable(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    out = x0.unsqueeze(1)
    return out


def sample_from_discretized_mix_logistic(l, nr_mix):
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda: temp = temp.cuda()
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    coeffs = torch.sum(F.tanh(
        l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.FloatTensor(means.size())
    if l.is_cuda: u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    u = Variable(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    x1 = torch.clamp(torch.clamp(
        x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, min=-1.), max=1.)
    x2 = torch.clamp(torch.clamp(
        x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, min=-1.), max=1.)

    out = torch.cat([x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1])], dim=3)
    # put back in Pytorch ordering
    out = out.permute(0, 3, 1, 2)
    return out


''' utilities for shifting the image around, efficient alternative to masking convolutions '''


def down_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.size()]
    # when downshifting, the last row is removed
    x = x[:, :, :xs[2] - 1, :]
    # padding left, padding right, padding top, padding bottom
    pad = nn.ZeroPad2d((0, 0, 1, 0)) if pad is None else pad
    return pad(x)


def right_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.size()]
    # when righshifting, the last column is removed
    x = x[:, :, :, :xs[3] - 1]
    # padding left, padding right, padding top, padding bottom
    pad = nn.ZeroPad2d((1, 0, 0, 0)) if pad is None else pad
    return pad(x)


def log_mix_dep_Logistic_256(x, params, average=False, n_comps=10):

    bin_size = 1. / 255.
    logits = params[:, 0:n_comps, :, :]
    means_r = params[:, n_comps:2 * n_comps, :, :]
    means_g = params[:, 2 * n_comps:3 * n_comps, :, :] + torch.tanh(params[:, 3 * n_comps:4 * n_comps]) * x[:, 0:1,
                                                                                                          :, :]
    means_b = params[:, 4 * n_comps:5 * n_comps, :, :] + torch.tanh(params[:, 5 * n_comps:6 * n_comps]) * x[:, 0:1,
                                                                                                          :, :] + \
              torch.tanh(params[:, 6 * n_comps:7 * n_comps, :, :]) * x[:, 1:2, :, :]

    log_scale_r = torch.clamp(params[:, 7 * n_comps:8 * n_comps, :, :], min=-7.)
    log_scale_g = torch.clamp(params[:, 8 * n_comps:9 * n_comps, :, :], min=-7.)
    log_scale_b = torch.clamp(params[:, 9 * n_comps:10 * n_comps, :, :], min=-7.)

    # final size is [B, N_comps, H, W, C]
    means = torch.cat([means_r[:, :, :, :, None], means_g[:, :, :, :, None], means_b[:, :, :, :, None]], 4)
    logvars = torch.cat(
        [log_scale_r[:, :, :, :, None], log_scale_g[:, :, :, :, None], log_scale_b[:, :, :, :, None]], 4)
    # final size is [B, C, H, W, N_comps]
    means = means.transpose(4, 1)
    logvars = logvars.transpose(4, 1)
    x = x[:, :, :, :, None]

    # calculate log probs per component
    # inv_scale = torch.exp(- logvar)[:, :, :, :, None]
    inv_scale = torch.exp(- logvars)
    centered_x = x - means
    inp_cdf_plus = inv_scale * (centered_x + .5 * bin_size)
    inp_cdf_minus = inv_scale * (centered_x - .5 * bin_size)
    cdf_plus = torch.sigmoid(inp_cdf_plus)
    cdf_minus = torch.sigmoid(inp_cdf_minus)

    # bin for 0 pixel is from -infinity to x + 0.5 * bin_size
    log_cdf_zero = F.logsigmoid(inp_cdf_plus)  # cdf_plus
    # bin for 255 pixel is from x - 0.5 * bin_size till infinity
    log_cdf_one = F.logsigmoid(- inp_cdf_minus)  # 1. - cdf_minus

    # calculate final log-likelihood for an image
    mask_zero = (x.data == 0.).float()
    mask_one = (x.data == 1.).float()

    log_logist_256 = mask_zero * log_cdf_zero + (1 - mask_zero) * mask_one * log_cdf_one + \
                     (1 - mask_zero) * (1 - mask_one) * torch.log(cdf_plus - cdf_minus + 1e-7)
    # [B, H, W, n_comps]
    log_logist_256 = torch.sum(log_logist_256, 1) + F.log_softmax(logits.permute(0, 2, 3, 1), 3)

    # log_sum_exp for n_comps
    log_logist_256 = log_sum_exp(log_logist_256)

    # flatten to [B, H * W]
    log_logist_256 = log_logist_256.view(log_logist_256.size(0), -1)

    # if reduce:
    if average:
        return torch.mean(log_logist_256, 1)
    else:
        return torch.sum(log_logist_256)
        # else:
    #     return log_logist_256


def sample(x_gen):
    n_comps = 10
    logits = x_gen[:, 0:n_comps, :, :]
    sel = torch.argmax(logits,  # -
                       # torch.log(- torch.log(self.float_tensor(logits.size()).uniform_(1e-5, 1-1e-5))),
                       dim=1, keepdim=True)
    one_hot = torch.zeros(logits.size())
    if torch.cuda.is_available():
        one_hot = one_hot.cuda()
    one_hot.scatter_(1, sel, 1.0)

    # log_scale_r = torch.sum(torch.clamp(x_gen[:, 7 * n_comps:8 * n_comps, :, :] +
    #                                     self.decoder.x_var[0, 0, 0, 0], min=-7.) * one_hot, 1, keepdim=True)
    # log_scale_g = torch.sum(torch.clamp(x_gen[:, 8 * n_comps:9 * n_comps, :, :] +
    #                                     self.decoder.x_var[0, 1, 0, 0], min=-7.) * one_hot, 1, keepdim=True)
    # log_scale_b = torch.sum(torch.clamp(x_gen[:, 9 * n_comps:10 * n_comps, :, :] +
    #                                     self.decoder.x_var[0, 2, 0, 0], min=-7.) * one_hot, 1, keepdim=True)

    mean_x_r = torch.sum(x_gen[:, n_comps:2 * n_comps, :, :] * one_hot, 1, keepdim=True)
    # u_r = self.float_tensor(mean_x_r.size()).uniform_(1e-5, 1 - 1e-5)
    x_r = F.hardtanh(mean_x_r,  # + torch.exp(log_scale_r) * (torch.log(u_r) - torch.log(1. - u_r)),
                     min_val=0., max_val=1.)

    mean_x_g = torch.sum(x_gen[:, 2 * n_comps:3 * n_comps, :, :] * one_hot, 1, keepdim=True) + \
               torch.tanh(torch.sum(x_gen[:, 3 * n_comps:4 * n_comps] * one_hot, 1, keepdim=True)) * x_r
    # u_g = self.float_tensor(mean_x_g.size()).uniform_(1e-5, 1 - 1e-5)
    x_g = F.hardtanh(mean_x_g,  # + torch.exp(log_scale_g) * (torch.log(u_g) - torch.log(1. - u_g)),
                     min_val=0., max_val=1.)

    mean_x_b = torch.sum(x_gen[:, 4 * n_comps:5 * n_comps, :, :] * one_hot, 1, keepdim=True) + \
               torch.tanh(torch.sum(x_gen[:, 5 * n_comps:6 * n_comps] * one_hot, 1, keepdim=True)) * x_r + \
               torch.tanh(
                   torch.sum(x_gen[:, 6 * n_comps:7 * n_comps, :, :] * one_hot, 1, keepdim=True)) * x_g
    # u_b = self.float_tensor(mean_x_b.size()).uniform_(1e-5, 1 - 1e-5)
    x_b = F.hardtanh(mean_x_b,  # + torch.exp(log_scale_b) * (torch.log(u_b) - torch.log(1. - u_b)),
                     min_val=0., max_val=1.)

    sample = torch.cat([x_r, x_g, x_b], 1)
    return sample