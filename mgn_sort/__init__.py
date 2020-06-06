from .deep_sort import DeepSort


__all__ = ['DeepSort', 'build_tracker']


def build_tracker(model, use_cuda):
    return DeepSort(model, use_cuda=use_cuda)
    









