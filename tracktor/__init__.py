from tracktor.oracle_tracker import OracleTracker
from tracktor.tracker import Tracker
from tracktor.reid.resnet import resnet50
from tracktor.utils import interpolate, plot_sequence, get_mot_accum, evaluate_mot_accums
import yaml
from tqdm import tqdm
import sacred
import torch


__all__ = ['Tracker', 'build_tracker']


def build_tracker():
    # reid
    reid_network = resnet50(pretrained=False)
    reid_network.load_state_dict(torch.load('/root/PycharmProjects/mmdet/tracktor/model/ResNet_iter_25245.pth',
                                            map_location=lambda storage, loc: storage))
    reid_network.eval()
    reid_network.cuda()
    f = open("/root/PycharmProjects/mmdet/tracktor/cfg/tracktor.yaml")
    tracktor = yaml.load(f)['tracktor']

    return Tracker(reid_network=reid_network, tracker_cfg=tracktor['tracker'])

# if __name__ == '__main__':
#     a = build_tracker()