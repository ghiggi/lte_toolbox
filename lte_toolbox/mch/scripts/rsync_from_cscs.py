import os
import subprocess
import numpy as np

year_str = '2022'

date_range = np.arange(152, 182) # June
date_range_str = ['%03d' % d for d in date_range]

username = 'aferrone'

data_path_base = '/store/msrad/radar/swiss/data/'

product_list = ['aZC', 'AZC', 'CPC', 'CPCH', 'RZC', 'LZC', 'MZC', 'BZC', 'CZC', 'EZC']

destination_dir_base = '/data/locarno/data_bgg/data'


# rsync -azv -e "ssh -A aferrone@ela.cscs.ch ssh" aferrone@tsa.cscs.ch:/store/msrad/radar/swiss/data/2022/22179/MLL22179.zip
command_copy_template = 'rsync -azv -e "ssh -A %s@ela.cscs.ch ssh" %s@tsa.cscs.ch:/store/msrad/radar/swiss/data/%s/%s/%s %s/%s'
fname_format = '%s%s.zip'

for curr_date in date_range_str:
    for curr_product in product_list:
        destination_dir_complete = os.path.join(destination_dir_base, year_str, year_str[2:]+curr_date)

        if not os.path.isdir(destination_dir_complete):
            os.makedirs(destination_dir_complete)

        fname = fname_format % (curr_product, year_str[2:]+curr_date)

        curr_command = command_copy_template % (username, username, year_str, year_str[2:]+curr_date, fname, destination_dir_complete, fname)

        os.system(curr_command)