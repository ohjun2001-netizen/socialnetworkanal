import json
import numpy as np
import pandas as pd

from research.data import ROOT
from research_confirmation.data import EXPECTED
from research_confirmation.features import boundaries,masks


def test_confirmation_was_frozen_before_data_and_has_success_rule():
    protocol=json.loads((ROOT/'research_confirmation'/'superuser_protocol.json').read_text())
    assert protocol['status_at_freeze'].startswith('No Super User event file')
    assert 'entirely above zero' in protocol['primary_success_rule']
    assert sum(EXPECTED.values())==1_443_339


def test_confirmation_partitions_are_mature_disjoint_and_daily_test():
    cutoffs=np.repeat(np.arange(0,200*86400,6*3600,dtype=np.int64),2)
    meta=pd.DataFrame({'cutoff':cutoffs,'label':np.tile([0,1],len(cutoffs)//2)})
    for boundary in boundaries(meta):
        part=masks(meta,boundary)
        assert not (sum(part.values())>1).any()
        assert (meta.loc[part['test'],'cutoff']%86400==0).all()
        assert meta.loc[part['train'],'cutoff'].max()+86400<=boundary['validation_start']
        assert meta.loc[part['tune'],'cutoff'].max()+86400<=boundary['calibration_start']
        assert meta.loc[part['cal'],'cutoff'].max()+86400<=boundary['test_start']
