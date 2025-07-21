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