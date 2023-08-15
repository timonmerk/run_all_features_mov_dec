import mne_bids
import mne
import numpy as np
from scipy import stats
import sys
import os

from bids import BIDSLayout

PATH_IN_BASE = r"C:\Users\ICN_admin\Documents\Datasets"

PATH_BIDS = os.path.join(PATH_IN_BASE, "Berlin")
layout = BIDSLayout(PATH_BIDS)

from pte import preprocessing
from pte import pipelines
from pte import filetools

run_files_Berlin = layout.get(
    #task=["SelfpacedRotationR", "SelfpacedRotationL", "SelfpacedForceWheel"],
    task=["SelfpacedHandTapL", "SelfpacedHandTapR", "SelfpacedRotationL", "SelfpacedRotationR", "SelfpacedForceWheel", "SelfpacedHandTapB"],
    extension=".vhdr",
)

run_files_Berlin = [f.path for f in run_files_Berlin]

PATH_RUN = run_files_Berlin[52]
bids_path = mne_bids.get_bids_path_from_fname(PATH_RUN)
raw_arr = mne_bids.read_raw_bids(bids_path)

#raw_arr.pick([ch for ch in raw_arr.ch_names if "ECOG" in ch or "SQUARED" in ch or "EMG" in ch])
#raw_arr.plot()

raw_emg = pipelines.process_emg_rms(raw_arr, emg_channels="EMG_R_BR_TM", window_duration=100, out_path=bids_path)

#out_path = PATH_BIDS

#raw = filetools.rewrite_bids_file(
#        raw=raw_emg, bids_path=out_path, reorder_channels=True
#)

raw_emg = preprocessing.emg.get_emg_rms(raw_arr, emg_ch="EMG_R_BR_TM", window_duration=100,)


