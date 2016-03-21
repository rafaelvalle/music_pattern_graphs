import os
import pretty_midi
import glob
from params import data_folder, missing_key_csv, broken_csv
from helpers import getImmediateSubdirectories, writeCSV

genres = getImmediateSubdirectories(data_folder)

gen_trk_dict = {}
for genre in genres:
    gen_trk_dict[genre] = glob.glob(os.path.join(data_folder, genre)+'/*.mid')

files_missing_key = []
broken_files = []
for genre in genres:
    for fullpath in gen_trk_dict[genre]:
        try:
            data = pretty_midi.PrettyMIDI(fullpath)
            if len(data.key_signature_changes) == 0:
                files_missing_key.append(fullpath)
        except:
            broken_files.append(fullpath)
            continue

writeCSV(missing_key_csv, files_missing_key)
writeCSV(broken_csv, broken_files)
