from __future__ import division
import os
import cPickle as pickle
import numpy as np
import glob
from helpers import getImmediateSubdirectories, openCSV
from params import data_folder
from params import missing_key_csv, broken_csv
from params import train_dic_name, test_dic_name
from params import train_test_ratio


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

np.random.seed(0)

genres = getImmediateSubdirectories(data_folder)
corrupted_files = openCSV(broken_csv)
missing_key_files = np.array(openCSV(missing_key_csv))

train_gen_trk_dict = {}
test_gen_trk_dict = {}

# prune corrupted files
for genre in genres:
    paths = [x for x in glob.glob(os.path.join(data_folder, genre)+'/*.mid')
             if [x] not in corrupted_files]
    ids = np.random.choice(paths, len(paths), replace=False)
    test_gen_trk_dict[genre] = paths[int(len(ids)*train_test_ratio):]
    train_gen_trk_dict[genre] = paths[:int(len(ids)*train_test_ratio)]

pickle.dump(train_gen_trk_dict, open(train_dic_name, 'wb'))
#pickle.dump(train_gen_trk_dict, open(test_dic_name, 'wb'))
pickle.dump(test_gen_trk_dict, open(test_dic_name, 'wb'))
