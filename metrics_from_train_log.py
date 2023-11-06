import re
from pathlib import Path
import argparse
import json

paperspace_files_DIR = Path("files_from_paperspace")
MODEL_CHECKPTS_DIR = paperspace_files_DIR / "model_checkpts" / "road" / "cache"
CHECKPTS_DIR = MODEL_CHECKPTS_DIR / "resnet50I3D512-Pkinetics-b2s4x1x1-roadt3-h3x3x3"
log_file_p_ = CHECKPTS_DIR / "train-10-27-20x.log"
EXTRACTED_METRICS_JSON_P_ = Path("swint_extracted_metrics.json")
MAX_METRIC_VAL_2_VAL_AND_EPOCH_NB_JSON_P_ = Path("swint_max_metric_2_val_and_epoch_nb.json")
EPOCH_SCORES_JSON_P_ = Path("swint_epoch_scores.json")

# Open the file for reading
# with open(log_file_p_.as_posix(), 'r') as log_file:
#     # Read the entire contents of the file into a string
#     log_contents = log_file.read()

# val_section_begin_delimiter = "Saving state,"
# val_section_end_delimiter = "Validation TIME"

# line_break = r"\n"

# epoch_NB_start = "epoch:"
# epoch_NB_pattern = re.compile(f'{epoch_NB_start}(.*?){line_break}')

# agentness_MEANAP_start = "agent_ness MEANAP:::=> "
# agentness_MEANAP_pattern = re.compile(f'{agentness_MEANAP_start}(.*?){line_break}')

# agent_MEANAP_start = "agent MEANAP:::=> "
# agent_MEANAP_pattern = re.compile(f'{agent_MEANAP_start}(.*?){line_break}')

# action_MEANAP_start = "action MEANAP:::=> "
# action_MEANAP_pattern = re.compile(f'{action_MEANAP_start}(.*?){line_break}')

# loc_MEANAP_start = "loc MEANAP:::=> "
# loc_MEANAP_pattern = re.compile(f'{loc_MEANAP_start}(.*?){line_break}')

# duplex_MEANAP_start = "duplex MEANAP:::=> "
# duplex_MEANAP_pattern = re.compile(f'{duplex_MEANAP_start}(.*?){line_break}')

# triplet_MEANAP_start = "triplet MEANAP:::=> "
# triplet_MEANAP_pattern = re.compile(f'{triplet_MEANAP_start}(.*?){line_break}')

# FRAME_Mean_AP_start = "FRAME Mean AP:: "
# FRAME_Mean_AP_pattern = re.compile(f'{FRAME_Mean_AP_start}(.*?){line_break}')

# ego_action_MEANAP_start = "ego_action MEANAP:::=> "
# ego_action_MEANAP_pattern = re.compile(f'{ego_action_MEANAP_start}(.*?){line_break}')

epochs_2_metrics = dict()
if EXTRACTED_METRICS_JSON_P_.exists():
    with open(EXTRACTED_METRICS_JSON_P_.as_posix(), 'r') as f_:
        epochs_2_metrics = json.load(f_)
else:
    EXTRACTED_METRICS_JSON_P_.touch()

# while True:
#     try:
#         index_val_begin = log_contents.index(val_section_begin_delimiter)
#         index_val_end = log_contents.index(val_section_end_delimiter)
#         val_section = log_contents[index_val_begin:index_val_end]
#         epoch_NB = epoch_NB_pattern.search(val_section).group(1)
#         #epoch_NB = int(epoch_NB)
#         epoch_metrics = dict()
#         agentness_MEANAP = agentness_MEANAP_pattern.search(val_section).group(1)
#         agentness_MEANAP = float(agentness_MEANAP)
#         epoch_metrics["agentness_MEANAP"] = agentness_MEANAP
#         agent_MEANAP = agent_MEANAP_pattern.search(val_section).group(1)
#         agent_MEANAP = float(agent_MEANAP)
#         epoch_metrics["agent_MEANAP"] = agent_MEANAP
#         action_MEANAP = action_MEANAP_pattern.search(val_section).group(1)
#         action_MEANAP = float(action_MEANAP)
#         epoch_metrics["action_MEANAP"] = action_MEANAP
#         loc_MEANAP = loc_MEANAP_pattern.search(val_section).group(1)
#         loc_MEANAP = float(loc_MEANAP)
#         epoch_metrics["loc_MEANAP"] = loc_MEANAP
#         duplex_MEANAP = duplex_MEANAP_pattern.search(val_section).group(1)
#         duplex_MEANAP = float(duplex_MEANAP)
#         epoch_metrics["duplex_MEANAP"] = duplex_MEANAP
#         triplet_MEANAP = triplet_MEANAP_pattern.search(val_section).group(1)
#         triplet_MEANAP = float(triplet_MEANAP)
#         epoch_metrics["triplet_MEANAP"] = triplet_MEANAP
#         FRAME_Mean_AP = FRAME_Mean_AP_pattern.search(val_section).group(1)
#         FRAME_Mean_AP = float(FRAME_Mean_AP)
#         epoch_metrics["FRAME_Mean_AP"] = FRAME_Mean_AP
#         ego_action_MEANAP = ego_action_MEANAP_pattern.search(val_section).group(1)
#         ego_action_MEANAP = float(ego_action_MEANAP)
#         epoch_metrics["ego_action_MEANAP"] = ego_action_MEANAP
#         epochs_2_metrics[epoch_NB] = epoch_metrics
#         log_contents = log_contents[index_val_end+1:]
#     except ValueError:
#         print(f"'{val_section_begin_delimiter}' not found in the string.")
#         break

compare_dict_keys_as_ints = lambda item: int(item[0])
with open(EXTRACTED_METRICS_JSON_P_.as_posix(), 'w') as f_:
    json.dump(dict(sorted(epochs_2_metrics.items(), key=compare_dict_keys_as_ints)), f_, indent=4)

METRICS = [
    "agentness_MEANAP", "agent_MEANAP", "action_MEANAP", "loc_MEANAP", 
    "duplex_MEANAP", "triplet_MEANAP", "FRAME_Mean_AP", "ego_action_MEANAP"
]
metric_2_maxval_and_epoch = dict()
if MAX_METRIC_VAL_2_VAL_AND_EPOCH_NB_JSON_P_.exists():
    with open(MAX_METRIC_VAL_2_VAL_AND_EPOCH_NB_JSON_P_.as_posix(), 'r') as f_:
        metric_2_maxval_and_epoch = json.load(f_)
else:
    MAX_METRIC_VAL_2_VAL_AND_EPOCH_NB_JSON_P_.touch()

for metric_key in METRICS:
    metric_max = None
    for epoch_nb in epochs_2_metrics.keys():
        epoch_metric = epochs_2_metrics[epoch_nb][metric_key]
        if metric_max is None or epoch_metric > metric_max:
            metric_max = epoch_metric
            metric_2_maxval_and_epoch[metric_key] = dict(max_val=metric_max, epoch=epoch_nb)

with open(MAX_METRIC_VAL_2_VAL_AND_EPOCH_NB_JSON_P_.as_posix(), 'w') as f_:
    json.dump(metric_2_maxval_and_epoch, f_, indent=4)

METRIC_TO_COEFFICIENT = dict(
    agentness_MEANAP=1, agent_MEANAP=1, action_MEANAP=1, loc_MEANAP=1,
    duplex_MEANAP=1, triplet_MEANAP=1, FRAME_Mean_AP=1, ego_action_MEANAP=0
)
SUM_METRIC_COEFFICIENTS = 0
for _, coeff in METRIC_TO_COEFFICIENT.items():
    SUM_METRIC_COEFFICIENTS += coeff

def calc_epoch_metrics_score(epoch_metrics: dict, metric_2_maxval_and_epoch: dict) -> float:
    accumulated_score = 0
    for metric in METRICS:
        epoch_metric = epoch_metrics[metric]
        metric_max_val_during_train = metric_2_maxval_and_epoch[metric]["max_val"]
        normalized_epoch_metric = epoch_metric / metric_max_val_during_train
        accumulated_score += METRIC_TO_COEFFICIENT[metric] * normalized_epoch_metric
    averaged_score = accumulated_score / SUM_METRIC_COEFFICIENTS
    return averaged_score

epoch_metric_scores = [
    (f"EPOCH: {epoch_nb}", f"SCORE: {calc_epoch_metrics_score(epochs_2_metrics[epoch_nb], metric_2_maxval_and_epoch)}")
    for epoch_nb in epochs_2_metrics.keys()
    ]

if not EPOCH_SCORES_JSON_P_.exists():
    EPOCH_SCORES_JSON_P_.touch()
with open(EPOCH_SCORES_JSON_P_.as_posix(), "w") as f_:
    json.dump(epoch_metric_scores, f_, indent=4)
