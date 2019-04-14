########################################################################
# Jason Baumbach
#   UCI Machine Learning Repository: Heart Disease Data Set
#       Convert the original Data Set into CSV files
#
#   Example of first instance from "hungarian.data" file:
#
#       1254 0 40 1 1 0 0
#       -9 2 140 0 289 -9 -9 -9
#       0 -9 -9 0 12 16 84 0
#       0 0 0 0 150 18 -9 7
#       172 86 200 110 140 86 0 0
#       0 -9 26 20 -9 -9 -9 -9
#       -9 -9 -9 -9 -9 -9 -9 12
#       20 84 0 -9 -9 -9 -9 -9
#       -9 -9 -9 -9 -9 1 1 1
#       1 1 -9. -9. name
#
# Source of the original files:
#   https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/
#
# The following websites were referenced:
#   Regular Expression HOWTO
#   https://docs.python.org/3/howto/regex.html
#
########################################################################

# required for determining data file name on the fly
import re
# required for outputing dictionary as CSV
import csv
# allow command line options
import argparse
parser = argparse.ArgumentParser(description="perform the k-means clustering on 1 to 2-dimensional data")
parser.add_argument("-f", "--filename", default="./data/hungarian.data", help="file name (and path if not in . dir)")
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=2, help="increase output verbosity")
args = parser.parse_args()

# parse input file name to use when creating the output file name
output_file = ''
# a regular expression to match the root of the file name supplied
match = re.search(r'([a-zA-Z0-9_-]+).data', args.filename)
# get the filename root
if match:
    if match.group(1):
        output_file = match.group(1) + '_orig.csv'
else:
    if args.verbosity > 0:
        print(f"Warning: no match for file name ({args.filename})")

if args.verbosity > 1:
    print(f"file root ({output_file})")

# typical header
header = ['id', 'ccf', 'age', 'sex', 'painloc', 'painexer', 'relrest', 'pncaden', 'cp', 'trestbps', 'htn', 'chol', 'smoke', 'cigs', 'years', 'fbs', 'dm', 'famhist', 'restecg', 'ekgmo', 'ekgday', 'ekgyr', 'dig', 'prop', 'nitr', 'pro', 'diuretic', 'proto', 'thaldur', 'thaltime', 'met', 'thalach', 'thalrest', 'tpeakbps', 'tpeakbpd', 'dummy', 'trestbpd', 'exang', 'xhypo', 'oldpeak', 'slope', 'rldv5', 'rldv5e', 'ca', 'restckm', 'exerckm', 'restef', 'restwm', 'exeref', 'exerwm', 'thal', 'thalsev', 'thalpul', 'earlobe', 'cmo', 'cday', 'cyr', 'num', 'lmt', 'ladprox', 'laddist', 'diag', 'cxmain', 'ramus', 'om1', 'om2', 'rcaprox', 'rcadist', 'lvx1', 'lvx2', 'lvx3', 'lvx4', 'lvf', 'cathef', 'junk', 'name']

if output_file == 'new_orig.csv':
    # long header for new.data file only
    header = ['id', 'ccf', 'age', 'sex', 'painloc', 'painexer', 'relrest', 'pncaden', 'cp', 'trestbps', 'htn', 'chol', 'smoke', 'cigs', 'years', 'fbs', 'dm', 'famhist', 'restecg', 'ekgmo', 'ekgday', 'ekgyr', 'dig', 'prop', 'nitr', 'pro', 'diuretic', 'proto', 'thaldur', 'thaltime', 'met', 'thalach', 'thalrest', 'tpeakbps', 'tpeakbpd', 'dummy', 'trestbpd', 'exang', 'xhypo', 'oldpeak', 'slope', 'rldv5', 'rldv5e', 'ca', 'restckm', 'exerckm', 'restef', 'restwm', 'exeref', 'exerwm', 'thal', 'thalsev', 'thalpul', 'earlobe', 'cmo', 'cday', 'cyr', 'num', 'lmt', 'ladprox', 'laddist', 'diag', 'cxmain', 'ramus', 'om1', 'om2', 'rcaprox', 'rcadist', 'lvx1', 'lvx2', 'lvx3', 'lvx4', 'lvf', 'cathef', 'junk', 'unk01' ,'unk02' ,'unk03' ,'unk04' ,'unk05' ,'unk06' ,'unk07' ,'unk08' ,'unk09' ,'unk10' ,'unk11' ,'unk12' ,'unk13' ,'unk14' ,'name']

# wrong header order...
#header = ['id', 'ccf', 'age', 'sex', 'painloc', 'painexer', 'trestbps', 'pncaden', 'cp', 'restbps', 'htn', 'chol', 'smoke', 'cigs', 'years', 'fbs', 'dm', 'famhist', 'restecg', 'ekgmo', 'ekgday', 'ekgyr', 'dig', 'prop', 'nitr', 'pro', 'diuretic', 'proto', 'thaldur', 'thaltime', 'met', 'thalach', 'thalrest', 'tpeakbps', 'tpeakbpd', 'dummy', 'trestbpd', 'exang', 'xhypo', 'oldpeak', 'slope', 'rldv5', 'rldv5e', 'ca', 'restckm', 'exerckm', 'restef', 'restwm', 'exeref', 'exerwm', 'thal', 'thalsev', 'thalpul', 'earlobe', 'cmo', 'cday', 'cyr', 'num', 'lmt', 'ladprox', 'laddist', 'diag', 'cxmain', 'ramus', 'om1', 'om2', 'rcaprox', 'rcadist', 'lvx1', 'lvx2', 'lvx3', 'lvx4', 'lvf', 'cathef', 'junk', 'name']
header_len = len(header)

# read in the 1-dimensional data file
with open(args.filename, mode='r') as data_file:
    record_accumulator = []  # list to hold line by line record accumulation
    record_dic = {} # dictionary to hold each of the records after accumulation
    rec_count = 0   # used as dictionary of records key
    skip_count = 0
    skip_rows = 0   # new.data contains duplicates of the cleveland.data
    if output_file == 'new_orig.csv':
        skip_rows = 280 # (i.e., the first 282 are cleveland.data duplicates)
    # parse data file
    for line in data_file:
        # remove real names
        if 'name' not in line:
            line = re.sub(r" [a-zA-Z]+", " name", line)
        record_accumulator += line.split()   # split line at every space character
        # last line of each record contains the word 'name'
        if 'name' in line:
            # skip the duplicate rows
            if skip_rows < skip_count:
                record_dic[rec_count] = record_accumulator   # add full record to the dictionary
                if header_len != len(record_accumulator):
                    if args.verbosity > 0:
                        print(f"Warning: header len({header_len}) != record len({len(record_accumulator)})")
                record_accumulator = []  # reset record accumulator to an empty list
                rec_count += 1
            else:
                record_accumulator = []  # reset record accumulator to an empty list
                skip_count += 1
        # if rec_count > 10:
        #     break

if args.verbosity > 1:
    print(f"records: {len(record_dic)}")
    for i in range(0, 1):
        print(f"rec {i}: {record_dic[i]}")

# On Windows, set newline='' to prevent from having extra carriage returns every line
#   https://stackoverflow.com/questions/3191528/csv-in-python-adding-an-extra-carriage-return-on-windows
with open('./data/' + output_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)
    for key, value in record_dic.items():
       writer.writerow(value)


# for row in range(0, record_dic):
#     for col in range(0, record_dic[row]):
#         print(f"rec [{row}][{col}]: {record_dic[row][col]}")


