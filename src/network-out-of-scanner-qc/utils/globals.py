SINGLE_TASKS_OUT_OF_SCANNER = [
    "cued_task_switching_single_task_network",
    "directed_forgetting_single_task_network",
    "flanker_single_task_network",
    "go_nogo_single_task_network",
    "n_back_single_task_network",
    "spatial_task_switching_single_task_network",
    "shape_matching_single_task_network",
    "stop_signal_single_task_network"
]

DUAL_TASKS_OUT_OF_SCANNER = [
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

SINGLE_TASKS_FMRI = [
    "cuedTS",
    "directedForgetting",
    "flanker",
    "goNogo",
    "nBack",
    "spatialTS",
    "shapeMatching",
    "stopSignal"
]

DUAL_TASKS_FMRI = [
    "directedForgettingWCuedTS",
    "directedForgettingWFlanker",
    "stopSignalWDirectedForgetting",
    "stopSignalWFlanker",
    "spatialTSWCuedTS",
    "flankerWShapeMatching",
    "cuedTSWFlanker",
    "spatialTSWShapeMatching",
    "nBackWShapeMatching",
    "nBackWSpatialTS"
]

FLANKER_CONDITIONS = ['congruent', 'incongruent']
DIRECTED_FORGETTING_CONDITIONS = ['con', 'pos', 'neg']
SPATIAL_TASK_SWITCHING_CONDITIONS = ['tstay_cstay', 'tstay_cswitch', 'tswitch_cswitch']
CUED_TASK_SWITCHING_CONDITIONS = ['tstay_cstay', 'tstay_cswitch', 'tswitch_cswitch']
N_BACK_CONDITIONS = []  # Will be generated dynamically based on trial_type and delay
STOP_SIGNAL_CONDITIONS = ['stop', 'go']
GO_NOGO_CONDITIONS = ['go', 'nogo']
SHAPE_MATCHING_CONDITIONS = ['SSS', 'SDD', 'DNN', 'DSD', 'DDS', 'DDD']
SPATIAL_WITH_CUED_CONDITIONS = ['cuedtstaycstay_spatialtstaycstay', 'cuedtstaycstay_spatialtstaycswitch', 
                                'cuedtstaycstay_spatialtswitchcswitch', 'cuedtstaycswitch_spatialtstaycstay',
                                'cuedtstaycswitch_spatialtstaycswitch', 'cuedtstaycswitch_spatialtswitchcswitch',
                                'cuedtswitchcswitch_spatialtstaycstay', 'cuedtswitchcswitch_spatialtstaycswitch',
                                'cuedtswitchcswitch_spatialtswitchcswitch']
FLANKER_WITH_CUED_CONDITIONS = ['congruent_tstay_cstay', 'congruent_tstay_cswitch', 'congruent_tswitch_new_cswitch', 'incongruent_tstay_cstay', 'incongruent_tstay_cswitch', 'incongruent_tswitch_new_cswitch']
GO_NOGO_WITH_CUED_CONDITIONS = ['go_tstay_cstay', 'go_tstay_cswitch', 'go_tswitch_cswitch', 'nogo_tstay_cstay', 'nogo_tstay_cswitch', 'nogo_tswitch_cswitch']
SHAPE_MATCHING_WITH_CUED_CONDITIONS = ['SSS_tstay_cstay', 'SSS_tstay_cswitch', 'SSS_tswitch_cswitch', 'SDD_tstay_cstay', 'SDD_tstay_cswitch', 'SDD_tswitch_cswitch', 'DNN_tstay_cstay', 'DNN_tstay_cswitch', 'DNN_tswitch_cswitch', 'DSD_tstay_cstay', 'DSD_tstay_cswitch', 'DSD_tswitch_cswitch', 'DDS_tstay_cstay', 'DDS_tstay_cswitch', 'DDS_tswitch_cswitch', 'DDD_tstay_cstay', 'DDD_tstay_cswitch', 'DDD_tswitch_cswitch']