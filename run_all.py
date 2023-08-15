import mne_bids
import mne
import numpy as np
from matplotlib import pyplot as plt
from py_neuromodulation import nm_IO, nm_define_nmchannels, nm_stream_offline
import py_neuromodulation as nm
from multiprocessing import Pool
from scipy import stats
import sys
import os

from joblib import Memory
from joblib import Parallel, delayed

from bids import BIDSLayout

PATH_IN_BASE = r"C:\Users\ICN_admin\Documents\Datasets"
PATH_OUT_BASE = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_new_all"

PATH_OUT_BASE = r"/data/gpfs-1/users/merkt_c/work/OUT/pynm_all_feat"
PATH_IN_BASE = r"/data/gpfs-1/users/merkt_c/work/Data_PyNm"
DEBUG = True

CHECK_IF_EXISTS = False

def est_features_run(PATH_RUN):

    def set_settings(settings: dict):

        settings["features"]["fft"] = True
        settings["features"]["fooof"] = True
        settings["features"]["return_raw"] = True
        settings["features"]["raw_hjorth"] = True
        settings["features"]["sharpwave_analysis"] = True
        settings["features"]["nolds"] = False
        settings["features"]["bursts"] = True
        settings["features"]["coherence"] = False

        settings["preprocessing"] = [
            "raw_resampling",
            "notch_filter",
            "re_referencing"
        ]

        settings["postprocessing"]["feature_normalization"] = True
        settings["postprocessing"]["project_cortex"] = False
        settings["postprocessing"]["project_subcortex"] = False

        return settings

    if any(x in PATH_RUN for x in ["Berlin", "Pittsburgh", "Beijing"]):
        if "Berlin" in PATH_RUN:
            PATH_BIDS = os.path.join(PATH_IN_BASE, "Berlin")
            PATH_OUT = os.path.join(PATH_OUT_BASE, "Berlin")
        elif "Beijing" in PATH_RUN:
            PATH_BIDS = os.path.join(PATH_IN_BASE, "Beijing")
            PATH_OUT = os.path.join(PATH_OUT_BASE, "Beijing")
        elif "Pittsburgh" in PATH_RUN:
            PATH_BIDS = os.path.join(PATH_IN_BASE, "Pittsburgh")
            PATH_OUT = os.path.join(PATH_OUT_BASE, "Pittsburgh")

        RUN_NAME = os.path.basename(PATH_RUN)[:-5]
        if os.path.exists(os.path.join(PATH_OUT, RUN_NAME)) is True and CHECK_IF_EXISTS is True:
            print("path exists")
            return
        (raw, data, sfreq, line_noise, _, _,) = nm_IO.read_BIDS_data(
            PATH_RUN=PATH_RUN, BIDS_PATH=PATH_BIDS, datatype="ieeg"
        )

        nm_channels = nm_define_nmchannels.set_channels(
            ch_names=raw.ch_names,
            ch_types=raw.get_channel_types(),
            reference="default",
            bads=raw.info["bads"],
            new_names="default",
            used_types=("ecog", "seeg", "dbs"),
            target_keywords=["SQUARED_EMG", "mov", "squared", "label", "SQUARED_ROTAWHEEL", "SQUARED_ROTATION", "rota_squared", "SQUARED_INTERPOLATED_EMG"],
        )

        if DEBUG is True:
            return nm_channels, data.shape[1]/sfreq

        settings = nm.nm_settings.get_default_settings()
        settings = nm.nm_settings.reset_settings(settings)

        settings = set_settings(settings)

        try:
            stream = nm.Stream(
                settings=settings,
                nm_channels=nm_channels,
                path_grids=None,
                verbose=True,
                sfreq=sfreq,
                line_noise=line_noise
            )

            stream.run(
                data=data[:, :10000],
                out_path_root=PATH_OUT,
                folder_name=RUN_NAME,
            )
        except:
            print(f"could not run {RUN_NAME}")
    else:
        PATH_OUT = os.path.join(PATH_OUT_BASE, "Washington")
        RUN_NAME = os.path.basename(PATH_RUN[:-4])  # cut .mat
        if os.path.exists(os.path.join(PATH_OUT, RUN_NAME)) is True and CHECK_IF_EXISTS is True:
            print("path exists")
            return
        dat = nm_IO.loadmat(PATH_RUN)
        label = dat["stim"]

        sfreq = 1000
        ch_names = [f"ECOG_{i}" for i in range(dat["data"].shape[1])]
        ch_names = ch_names + ["mov"]
        ch_types = ["ecog" for _ in range(dat["data"].shape[1])]
        ch_types = ch_types + ["misc"]
        data_uv = dat["data"] * 0.0298  # see doc description

        data = np.concatenate((data_uv, np.expand_dims(label, axis=1)), axis=1).T

        nm_channels = nm_define_nmchannels.set_channels(
            ch_names=ch_names,
            ch_types=ch_types,
            reference="default",
            bads=None,
            used_types=["ecog"],
            target_keywords=[
                "mov",
            ],
        )

        if DEBUG is True:
            return nm_channels, data.shape[1]/sfreq

        # read electrodes
        sub_name = RUN_NAME[:2]
        electrodes = (
            nm_IO.loadmat(
                os.path.join(
                    PATH_IN_BASE,
                    r"Washington\motor_basic\transform_mni",
                    sub_name + "_electrodes.mat",
                )
            )["mni_coord"]
            / 1000
        )  # transform into m

        stream = nm_stream_offline.Stream(
            settings=None,
            nm_channels=nm_channels,
            verbose=True,
        )

        stream.set_settings_fast_compute()
        stream.settings = set_settings(stream.settings)

        stream.init_stream(
            sfreq=sfreq,
            line_noise=60,
            coord_list=list(electrodes),
            coord_names=ch_names,
        )

        stream.nm_channels.loc[
            stream.nm_channels.query('type == "misc"').index, "target"
        ] = 1

        stream.run(
            data=data[:, :],
            out_path_root=PATH_OUT,
            folder_name=RUN_NAME,
        )

def collect_all_runs():

    # collect all run files
    PATH_BIDS = os.path.join(PATH_IN_BASE, "Berlin")
    layout = BIDSLayout(PATH_BIDS)


    # for the selfpacedHandTapL, there is only EMG, needs to be squared..
    run_files_Berlin = layout.get(
        #task=["SelfpacedRotationR", "SelfpacedRotationL", "SelfpacedForceWheel"],
        task=["SelfpacedHandTapL", "SelfpacedHandTapR", "SelfpacedRotationL", "SelfpacedRotationR", "SelfpacedForceWheel", "SelfpacedHandTapB"],
        extension=".vhdr",
    )

    run_files_Berlin = [f.path for f in run_files_Berlin]


    PATH_BIDS = os.path.join(PATH_IN_BASE, "Beijing")
    layout = BIDSLayout(PATH_BIDS)
    run_files_Beijing = layout.get(
        task=["ButtonPressL", "ButtonPressR"], extension=".vhdr"
    )
    run_files_Beijing = [f.path for f in run_files_Beijing]

    PATH_BIDS = os.path.join(PATH_IN_BASE, "Pittsburgh")
    layout = BIDSLayout(PATH_BIDS)
    run_files_Pittsburgh = layout.get(
        task=[
            "force",
        ],
        extension=".vhdr",
    )
    run_files_Pittsburgh = [f.path for f in run_files_Pittsburgh]

    PATH_DATA = os.path.join(PATH_IN_BASE, r"Washington\motor_basic\data")
    run_files_Washington = [
        os.path.join(PATH_DATA, f) for f in os.listdir(PATH_DATA) if "_mot_t_h" in f
    ]

    run_all = np.concatenate(
        [
            run_files_Berlin,
            run_files_Beijing,
            run_files_Pittsburgh,
            run_files_Washington,
        ]
    )

    return run_all

if __name__ == "__main__":

    run_idx = int(sys.argv[1])

    l_runs = collect_all_runs()

    Parallel(n_jobs=12)(
        delayed(est_features_run)(sub_cohort)
        for sub_cohort in l_runs[run_idx:run_idx+12]
    )

    # check how many runs exist, then separate those across the tasks
    
    #nm_duration = []
    #for run in l_runs:
    #    _, duration = est_features_run(run)
    #    nm_duration.append(duration)

    #est_features_run(l_runs[0])

    #pool = Pool(processes=12)
    #pool.map(est_features_run, run_all[run_idx:run_idx+12])

    #nm_ch = []
    #for run in run_all:
    #    nm_ch.append(est_features_run(run))
    
    # check for each dataframe in nm_ch if at least one row has target column == 1
    # if not, then delete the dataframe
    #for idx, df in enumerate(nm_ch):
    #    if df.query("target == 1").shape[0] == 0:
    #        print(f" {run_all[idx]} has no target")


    # run parallel Pool

    # def debug():
    #     #PATH_RUN = run_files_Berlin[63]
    #     #bids_path = mne_bids.get_bids_path_from_fname(PATH_RUN)
    # 
    #     # check first if all paths can be read!
    #     for idx, PATH_RUN in enumerate(run_files_Berlin):
    #         print(PATH_RUN)
    #         if idx == 53:
    #             print()
    #         print(f"index: {idx}")
    #         bids_path = mne_bids.get_bids_path_from_fname(PATH_RUN)
    #         raw_arr = mne_bids.read_raw_bids(bids_path)
    # 
    # 
    #     run_files_Berlin = layout.get(
    #         task=["SelfpacedHandTapB"],
    #         extension=".vhdr",
    #     )
    #     
    #     raw_arr.pick(["ECOG_R_01_SMC_AT", "ECOG_R_02_SMC_AT", "ECOG_R_03_SMC_AT", "ECOG_R_04_SMC_AT", "ECOG_R_05_SMC_AT",
    #                 "ECOG_R_06_SMC_AT", "EMG_R_BR_TM", "EMG_L_BR_TM"])
    #     
    #     raw_arr.pick([ch for ch in raw_arr.ch_names if "ECOG" in ch or "SQUARED" in ch or "EMG" in ch])
    #     raw_arr.plot(block=True)
    #     #est_features_run(run_files_Berlin[10])
    # 
    #     l_ch = []
    #     for PATH_RUN in run_files_Beijing:
    #         raw = mne.io.read_raw_brainvision(PATH_RUN)
    #         l_ch.append(raw.ch_names) 
    #     raw_arr = mne.io.read_raw_brainvision(run_files_Beijing[3])
