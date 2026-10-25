import numpy as np
import pandas as pd
from research_extensions.error_analysis import confusion,next_delays,future_activity


def test_error_denominators_and_empty_predictions():
    f=pd.DataFrame({'label':[1,1,1,0,0,0,0],'prediction':[1,1,0,1,0,0,0]})
    r=confusion(f)
    assert (r['tp'],r['fn'],r['fp'],r['tn'])==(2,1,1,3)
    assert r['miss_rate']==1/3 and r['false_discovery_rate']==1/3
    assert r['false_positive_rate']==1/4
    f['prediction']=False
    assert confusion(f)['precision'] is None


def test_future_timing_exact_boundaries_and_censoring():
    e=pd.DataFrame({'src':[0,0,0,1],'dst':[1,1,1,2],'time':[0,86400,172800,200000]})
    c=pd.DataFrame({'src':[0,0,1],'dst':[1,1,2],'cutoff':[0,1,199999]})
    delay,complete=next_delays(e,c)
    np.testing.assert_allclose(delay,[0,86399/3600,1/3600])
    assert complete.tolist()==[True,True,False]


def test_future_activity_is_directional_and_half_open():
    e=pd.DataFrame({'src':[0,0,1,2],'dst':[1,2,2,1],'time':[0,86400,86401,172800]})
    c=pd.DataFrame({'src':[0,0,1],'dst':[1,1,2],'cutoff':[1,86401,1]})
    assert future_activity(e,c,'src').tolist()==[True,False,False]
    assert future_activity(e,c,'dst').tolist()==[False,True,True]
