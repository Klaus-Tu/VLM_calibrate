import torch
import numpy as np

def tune_temp(logits, labels, binary_search=True, lower=0.5, upper=10, eps=0.0001):
    logits = np.array(logits)

    if binary_search:
        import torch
        import torch.nn.functional as F

        logits = torch.FloatTensor(logits)
        labels = torch.LongTensor(labels)
        t_guess = torch.FloatTensor([0.5*(lower + upper)]).requires_grad_()

        while upper - lower > eps:
            if torch.autograd.grad(F.cross_entropy(logits / t_guess, labels), t_guess)[0] > 0:
                upper = 0.5 * (lower + upper)
            else:
                lower = 0.5 * (lower + upper)
            t_guess = t_guess * 0 + 0.5 * (lower + upper)

        t = min([lower, 0.5 * (lower + upper), upper], key=lambda x: float(F.cross_entropy(logits / x, labels)))
    else:
        import cvxpy as cx

        set_size = np.array(logits).shape[0]

        t = cx.Variable()

        expr = sum((cx.Minimize(cx.log_sum_exp(logits[i, :] * t) - logits[i, labels[i]] * t)
                    for i in range(set_size)))
        p = cx.Problem(expr, [lower <= t, t <= upper])

        p.solve()   # p.solve(solver=cx.SCS)
        t = 1 / t.value

    return t


class _ECELoss():
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def eval(self, confidences, accuracies):
        ece = np.zeros(1)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.__gt__(bin_lower) * confidences.__le__(bin_upper)
            prop_in_bin = in_bin.astype(float).mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].astype(float).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


def ECE(logits, labels, n_bins=15, isLogits=0):
    # logits : logits of model's prediction (n_sample * n_class)
    # labels : ground truth (n_samples, )
    ece_criterion = _ECELoss(n_bins)
    if isLogits == 0:
        softmaxes = torch.nn.functional.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
    elif isLogits == 1:
        # input is softmax(logits)
        softmaxes = logits
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
    elif isLogits == 2:
        # input is confidences-->argmax(softmax(logits))
        # ipdb.set_trace()
        # confidences = logits
        # accuracies = labels
        acc = labels.sum() / len(labels)
        ece = ece_criterion.eval(logits.cpu().numpy(), labels.cpu().numpy())
        return acc, ece

    acc = accuracies.float().sum() / len(accuracies)
    ece = ece_criterion.eval(confidences.cpu().numpy(), accuracies.cpu().numpy())
    return acc, ece
