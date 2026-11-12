import torch

from models import CrossAug


class Exp_Basic:
    """Preserve the experiment base class responsibilities: model registration, device selection, and model construction."""

    def __init__(self, args):
        self.args = args
        self.model_dict = {'CrossAug': CrossAug}
        if args.use_dp:
            self.device = self._acquire_device()
            self.model = self._build_model().to(self.device)
        else:
            self.device = torch.device('cuda:{}'.format(args.local_rank))
            self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        device = torch.device('cuda:{}'.format(self.args.device_ids[0]))
        print('Use GPU:', device)
        return device
