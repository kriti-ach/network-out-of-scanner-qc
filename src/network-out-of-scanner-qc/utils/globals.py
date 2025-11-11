SINGLE_TASKS = [
    "cued_task_switching_single_task_network",
    "directed_forgetting_single_task_network",
    "flanker_single_task_network",
    "go_nogo_single_task_network",
    "n_back_single_task_network",
    "spatial_task_switching_single_task_network",
    "shape_matching_single_task_network",
    "stop_signal_single_task_network"
]

DUAL_TASKS = [
    "cued_task_switching_with_directed_forgetting",
    "directed_forgetting_with_flanker",
    "directed_forgetting_with_shape_matching",
    "flanker_with_shape_matching",
    "flanker_with_cued_task_switching",
    "flanker_with_spatial_task_switching",
    "go_nogo_with_cued_task_switching",
    "go_nogo_with_directed_forgetting",
    "go_nogo_with_flanker",
    "go_nogo_with_n_back",
    "go_nogo_with_shape_matching",
    "go_nogo_with_spatial_task_switching",
    "n_back_with_cued_task_switching",
    "n_back_with_directed_forgetting",
    "n_back_with_flanker",
    "n_back_with_shape_matching",
    "n_back_with_spatial_task_switching",
    "shape_matching_with_cued_task_switching",
    "shape_matching_with_spatial_task_switching",
    "spatial_task_switching_with_cued_task_switching",
    "spatial_task_switching_with_directed_forgetting",
    "stop_signal_with_cued_task_switching",
    "stop_signal_with_directed_forgetting",
    "stop_signal_with_flanker",
    "stop_signal_with_n_back",
    "stop_signal_with_shape_matching",
    "stop_signal_with_spatial_task_switching",
    "stop_signal_with_go_nogo"
]


FLANKER_CONDITIONS = ['congruent', 'incongruent']
DIRECTED_FORGETTING_CONDITIONS = ['con', 'pos', 'neg']
SPATIAL_TASK_SWITCHING_CONDITIONS = ['tstay_cstay', 'tstay_cswitch', 'tswitch_cswitch']
CUED_TASK_SWITCHING_CONDITIONS = ['tstay_cstay', 'tstay_cswitch', 'tswitch_cswitch']
N_BACK_CONDITIONS = []  # Will be generated dynamically based on trial_type and delay
GO_NOGO_CONDITIONS = ['go', 'nogo']
SHAPE_MATCHING_CONDITIONS = ['SSS', 'SDD', 'DNN', 'DSD', 'DDS', 'DDD', 'SNN']
SPATIAL_WITH_CUED_CONDITIONS = ['cuedtstaycstay_spatialtstaycstay', 'cuedtstaycstay_spatialtstaycswitch', 
                                'cuedtstaycstay_spatialtswitchcswitch', 'cuedtstaycswitch_spatialtstaycstay',
                                'cuedtstaycswitch_spatialtstaycswitch', 'cuedtstaycswitch_spatialtswitchcswitch',
                                'cuedtswitchcswitch_spatialtstaycstay', 'cuedtswitchcswitch_spatialtstaycswitch',
                                'cuedtswitchcswitch_spatialtswitchcswitch']
FLANKER_WITH_CUED_CONDITIONS = ['congruent_tstay_cstay', 'congruent_tstay_cswitch', 'congruent_tswitch_new_cswitch', 'incongruent_tstay_cstay', 'incongruent_tstay_cswitch', 'incongruent_tswitch_new_cswitch']
GO_NOGO_WITH_CUED_CONDITIONS = ['go_tstay_cstay', 'go_tstay_cswitch', 'go_tswitch_cswitch', 'nogo_tstay_cstay', 'nogo_tstay_cswitch', 'nogo_tswitch_cswitch']
SHAPE_MATCHING_WITH_CUED_CONDITIONS = ['SSS_tstay_cstay', 'SSS_tstay_cswitch', 
                                'SSS_tswitch_new_cswitch', 'SDD_tstay_cstay', 'SDD_tstay_cswitch', 
                                'SDD_tswitch_new_cswitch', 'DNN_tstay_cstay', 'DNN_tstay_cswitch', 
                                'DNN_tswitch_new_cswitch', 'DSD_tstay_cstay', 'DSD_tstay_cswitch', 
                                'DSD_tswitch_new_cswitch', 'DDS_tstay_cstay', 'DDS_tstay_cswitch', 
                                'DDS_tswitch_new_cswitch', 'DDD_tstay_cstay', 'DDD_tstay_cswitch', 
                                'DDD_tswitch_new_cswitch', 'SNN_tstay_cstay', 'SNN_tstay_cswitch', 
                                'SNN_tswitch_new_cswitch']
CUED_TASK_SWITCHING_WITH_DIRECTED_FORGETTING_CONDITIONS = ['con_tstay_cstay', 'con_tstay_cswitch', 'con_tswitch_cswitch', 'pos_tstay_cstay', 'pos_tstay_cswitch', 'pos_tswitch_cswitch', 'neg_tstay_cstay', 'neg_tstay_cswitch', 'neg_tswitch_cswitch']
SHAPE_MATCHING_CONDITIONS_WITH_DIRECTED_FORGETTING = ['match', 'mismatch']
FLANKER_WITH_CUED_CONDITIONS_FMRI = ['congruent_tstay_cstay', 'congruent_tstay_cswitch', 'congruent_tswitch_cswitch', 'incongruent_tstay_cstay', 'incongruent_tstay_cswitch', 'incongruent_tswitch_cswitch']

# Stop signal task
STOP_SUCCESS_ACC_LOW_THRESHOLD = 0.25
STOP_SUCCESS_ACC_HIGH_THRESHOLD = 0.75
GO_RT_THRESHOLD = 850
GO_RT_THRESHOLD_DUAL_TASK = 1000
GO_RT_THRESHOLD_FMRI = 1000

#Go nogo task
GO_ACC_THRESHOLD_GO_NOGO = 0.8
NOGO_ACC_THRESHOLD_GO_NOGO = 0.2
GO_OMISSION_RATE_THRESHOLD = 0.25
NOGO_STOP_SUCCESS_MIN = 0.2

#N back task
MISMATCH_THRESHOLD = 0.55
MATCH_THRESHOLD = 0.2
MISMATCH_COMBINED_THRESHOLD = 0.7
MATCH_COMBINED_THRESHOLD = 0.55

# N-back fMRI exclusion thresholds (both conditions must be met)
NBACK_1BACK_MATCH_ACC_COMBINED_THRESHOLD_1 = 0.2
NBACK_1BACK_MISMATCH_ACC_COMBINED_THRESHOLD_1 = 0.75
NBACK_1BACK_MATCH_ACC_COMBINED_THRESHOLD_2 = 0.5
NBACK_1BACK_MISMATCH_ACC_COMBINED_THRESHOLD_2 = 0.5
NBACK_2BACK_MATCH_ACC_COMBINED_THRESHOLD_1 = 0.2
NBACK_2BACK_MISMATCH_ACC_COMBINED_THRESHOLD_1 = 0.75
NBACK_2BACK_MATCH_ACC_COMBINED_THRESHOLD_2 = 0.5
NBACK_2BACK_MISMATCH_ACC_COMBINED_THRESHOLD_2 = 0.5

# Go_nogo fMRI exclusion thresholds (both conditions must be met)
GONOGO_GO_ACC_THRESHOLD_1 = 0.75
GONOGO_NOGO_ACC_THRESHOLD_1 = 0.2
GONOGO_GO_ACC_THRESHOLD_2 = 0.5
GONOGO_NOGO_ACC_THRESHOLD_2 = 0.5

# All other tasks
ACC_THRESHOLD = 0.55
OMISSION_RATE_THRESHOLD = 0.25

SUMMARY_ROWS = 4
LAST_N_TEST_TRIALS = 10