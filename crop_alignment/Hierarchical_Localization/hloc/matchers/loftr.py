import torch
import warnings
from kornia.feature.loftr.loftr import default_cfg
from kornia.feature import LoFTR as LoFTR_
from kornia.utils.helpers import map_location_to_cpu


from ..utils.base_model import BaseModel

from typing import Dict

urls: Dict[str, str] = {}
urls["loftr_23_0.5"] = "https://polybox.ethz.ch/index.php/s/4VVcO0oAXvz8bdo/download"
urls["loftr_23_0.5_hc"] = "https://polybox.ethz.ch/index.php/s/jzGrkJyyj6g8hiM/download"

class LoFTR(BaseModel):
    default_conf = {
        'weights': 'outdoor',
        'match_threshold': 0.2,
        'max_num_matches': None,
    }
    required_inputs = [
        'image0',
        'image1'
    ]

    def _init(self, conf):
        cfg = default_cfg
        cfg['match_coarse']['thr'] = conf['match_threshold']
        # self.net = LoFTR_(pretrained=conf['weights'], config=cfg)
        try:
            self.net = LoFTR_(pretrained=conf['weights'], config=cfg)
        except ValueError:
            self.net = LoFTR_(pretrained=None, config=cfg)
            pretrained_dict = torch.hub.load_state_dict_from_url(urls[conf['weights']], map_location=map_location_to_cpu, file_name=conf['weights']+'.ckpt')
            self.net.load_state_dict(pretrained_dict['state_dict'])
            self.net.eval()

    def _forward(self, data):
        # For consistency with hloc pairs, we refine kpts in image0!
        rename = {
            'keypoints0': 'keypoints1',
            'keypoints1': 'keypoints0',
            'image0': 'image1',
            'image1': 'image0',
            'mask0': 'mask1',
            'mask1': 'mask0',
        }
        data_ = {rename[k]: v for k, v in data.items()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self.net(data_)

        scores = pred['confidence']

        top_k = self.conf['max_num_matches']
        if top_k is not None and len(scores) > top_k:
            keep = torch.argsort(scores, descending=True)[:top_k]
            pred['keypoints0'], pred['keypoints1'] =\
                pred['keypoints0'][keep], pred['keypoints1'][keep]
            scores = scores[keep]

        # Switch back indices
        pred = {(rename[k] if k in rename else k): v for k, v in pred.items()}
        pred['scores'] = scores
        del pred['confidence']
        return pred
