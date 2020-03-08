import numpy as np
import _pickle as pickle

'''
Drawing each individual image one by one to confirm the quality of data
'''

with open(r'48net\48\cls0.imdb','rb') as fid:
    cls = pickle.load(fid)
# with open(r'48net\48\pts.imdb', 'rb') as fid:
#     pts = pickle.load(fid)
with open(r'48net\48\roi0.imdb', 'rb') as fid:
    roi = pickle.load(fid)
ims_cls = []
# ims_pts = []
ims_roi = []
cls_score = []
# pts_score = []
roi_score = []
for (idx, dataset) in enumerate(cls) :
    ims_cls.append( np.swapaxes(dataset[0],0,2))
    cls_score.append(dataset[1])
for (idx,dataset) in enumerate(roi) :
    ims_roi.append( np.swapaxes(dataset[0],0,2))
    roi_score.append(dataset[2])
# for (idx,dataset) in enumerate(pts) :
#     ims_pts.append( np.swapaxes(dataset[0],0,2))
#     pts_score.append(dataset[3])

ims_cls = np.array(ims_cls)
# ims_pts = np.array(ims_pts)
ims_roi = np.array(ims_roi)
cls_score = np.array(cls_score)
# pts_score = np.array(pts_score)
roi_score = np.array(roi_score)
