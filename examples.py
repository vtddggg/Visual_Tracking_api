from Sequence import Sequence
from tools import Tracking,visulize_result

sequence = Sequence(path='/media/maoxiaofeng/project/GameProject/dataset/vot-tir2016', name='quadrocopter', region_format='rectangle')

Tracking(sequence,tracker_list=['KCFtracker','DSSTtracker'],visualize=False)
#visulize_result(sequence,tracker_list=['KCFtracker','DSSTtracker'],visualize_gt = True)