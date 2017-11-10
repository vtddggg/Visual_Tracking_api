from Sequence import Sequence
from tools import Tracking,visulize_result,precision_plot,overlap_plot

sequence = Sequence(path='/media/maoxiaofeng/project/GameProject/dataset/vot-tir2016', name='birds', region_format='rectangle')

#Tracking(sequence,tracker_list=['HCFtracker'],visualize=False)
#visulize_result(sequence,tracker_list=['HCFtracker'],visualize_gt = True)
#precision_plot(sequence, tracker_list=['HCFtracker'])
#overlap_plot(sequence, tracker_list=['HCFtracker'])