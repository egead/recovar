import os
import obspy
from continuous_data_processor_fast import ContinuousDataPreprocessor

#catalogs_dir = '/home/boxx/Public/earthquake_model_evaluations/data/SilivriPaper_2019-09-01__2019-11-30/processed_catalogs'
#kara74_cat = os.path.join(catalogs_dir, 'kara74a_phase_picks.csv')
#waveforms_dir = "/home/boxx/Public/earthquake_model_evaluations/data/SilivriPaper_2019-09-01__2019-11-30/prepared_waveforms/day_by_day"
catalog="/mnt/second_drive/recovar/reproducibility/catalog_balikesir_recovar_2025-07-01_2026-02-28.csv"
waveforms_dir="/mnt/second_drive/recovar-balikesir/data/continuous/"

station_dirs = []
for (root, dirs, files) in os.walk(waveforms_dir):
    for dir in dirs:
        station_dirs.append(os.path.join(waveforms_dir,dir)) 


preprocessor = ContinuousDataPreprocessor(
    catalog_csv=catalog,
    output_hdf5_path=f"output/BALIKESIR_continuous_waveforms.hdf5",
    output_metadata_csv_path=f"output/BALIKESIR_continuous_metadata.csv",
    window_length=60, 
    sampling_rate=100
)
for station in station_dirs:
	preprocessor.process_station(station)
