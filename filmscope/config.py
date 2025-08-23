# location where data was downloaded
import os
path_to_data = "/home/steven/Documents/FiLMScope-sjt/FiLMScope_paper_data"
alt_path = "/data2/steven"
# path_to_data = "/media/Friday/Temporary/Clare/20241211_calibration_tests"
# path_to_data = "/media/Friday/Temporary/Clare/20241226_fluoro_chicken"
# path_to_data = "/media/Friday/Temporary/Clare/20250116_macaque_brain"

# this is where logging will be performed by some scripts
log_folder = path_to_data + '/log_folder'

# project and api_token can be generated at neptune.ai
# but are not necessary to run the example scripts
neptune_project = ""
neptune_api_token = ""
