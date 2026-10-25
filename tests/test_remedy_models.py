import numpy as np
import pandas as pd

from research.features import NAMES
from research_extensions.remedy_models import activity_view, expert_group


def test_activity_target_and_source_rows_are_leakage_safe():
    events=pd.DataFrame([(0,1,0,'a2q'),(0,2,20,'c2q'),(1,2,100,'c2a'),
                         (0,3,86410,'c2a'),(1,3,86500,'a2q')],columns=['src','dst','time','kind'])
    meta=pd.DataFrame({'cutoff':[10,10,10,10,200,200], 'src':[0,0,1,1,0,1],
                       'dst':[1,2,2,3,1,2], 'label':[0,0,0,0,0,0]})
    x=np.zeros((len(meta),len(NAMES)),np.float32)
    unique,ax,ay,group,pair,first=activity_view(meta,x,events)
    assert unique.to_dict('records')==[{'cutoff':10,'src':0},{'cutoff':10,'src':1},
                                       {'cutoff':200,'src':0},{'cutoff':200,'src':1}]
    assert ay.tolist()==[1,1,1,1] and pair.tolist()==[1]*6
    np.testing.assert_allclose(ax[:,-1],[np.log1p(2),np.log1p(2),np.log1p(1),np.log1p(1)],rtol=1e-6)
    changed=events.copy();changed.loc[changed.time>=10,'dst']+=5
    _,xx,yy,_,_,_=activity_view(meta,x,changed)
    np.testing.assert_array_equal(ax,xx)
    np.testing.assert_array_equal(ay,yy)


def test_expert_groups_cross_recency_and_last_type():
    x=np.zeros((6,len(NAMES)),np.float32)
    x[:,NAMES.index('hours_since_pair_event')]=[1,1,1,12,12,12]
    for row,kind in enumerate(('a2q','c2a','c2q','a2q','c2a','c2q')):
        x[row,NAMES.index(f'type_{kind}_last')]=1
    assert expert_group(x).tolist()==list(range(6))
