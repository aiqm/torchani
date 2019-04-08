def enable(accelerator='all'):
    if accelerator == 'all' or accelerator == 'aev.triple_by_molecule':
        pass
    else:
        raise ValueError('Unknown Accelerator')
