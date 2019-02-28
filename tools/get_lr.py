def get_learning_rate(optimizer):
    """
    Gets Learning Rate from Pytorch Optimizer.
    
    Parameters:
    optimizer - Pytorch Optimizer

    lr (list) - value of lr param in optimizer
    """
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr
