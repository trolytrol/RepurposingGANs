
# For old version pytorch
def softmax_cross_entropy_with_softtarget(input, target, mask, reduction='mean'):
        """
        :param input: (batch, *)
        :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
        """
        input = input.view((input.shape[0], input.shape[1], -1))
        target = target.view((target.shape[0], target.shape[1], -1))
        mask = mask.view((mask.shape[0], -1)).unsqueeze(1)

        logprobs = torch.nn.functional.log_softmax(input, dim=1)
        loss = - torch.sum((mask*target) * logprobs, dim=1)
        batchloss = torch.sum(loss, dim=1)/torch.sum(mask, dim=(1,2))

        if reduction == 'none':
            return batchloss
        elif reduction == 'mean':
            return torch.mean(batchloss)
        elif reduction == 'sum':
            return torch.sum(batchloss)
        else:
            raise NotImplementedError('Unsupported reduction mode.')