########################################################################
# Jason Baumbach
#   Read in text file, clean up contents, then output new text file
########################################################################

# required for reading csv files to get just the header
import csv
# required for reading text file into numpy matrix
import numpy as np
# required for determining data file name on the fly
import re
# allow command line options
import argparse
# used to normalize so I don't have to reinvent "this" wheel (it's already 04/10/2019)
#from sklearn.preprocessing import normalize

# code adopted from:
#   https://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-between-0-0-1-0-usin
def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

parser = argparse.ArgumentParser(description="perform the k-means clustering on 1 to 2-dimensional data")
parser.add_argument("-f", "--filename", default="./data/hungarian_orig.csv", help="file name (and path if not in . dir)")
parser.add_argument("-ct", "--columntolerance", type=restricted_float, default=0.0, help="tolerance percentage as a decimal for missing column data")
parser.add_argument("-rt", "--rowtolerance", type=restricted_float, default=0.0, help="tolerance percentage as a decimal for missing row data")
parser.add_argument("-cs", "--columnset", choices=['uci', 'uci+', 'min', 'max'], default='max', help="limit the attributes to be considered")
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=2, help="increase output verbosity")
args = parser.parse_args()

# parse input file name to use when creating the output file name
local_dic = {
    'file_root' : ''
    , 'file_dir' : './data/'
    , 'target_col_name' : 'num'
    , 'target_col_index' : -1
    # used to create new "smoke" column as (years/age*cigs)
    , 'age_col_index' : -1
    , 'cigs_col_index' : -1
    , 'years_col_index' : -1
    # used to reorder to "cp" column
    , 'cp_col_index' : -1
    # Various Headers
    , 'file_header_92' : ['id', 'ccf', 'age', 'sex', 'painloc', 'painexer', 'relrest', 'pncaden', 'cp', 'trestbps', 'htn', 'chol', 'smoke', 'cigs', 'years', 'fbs', 'dm', 'famhist', 'restecg', 'ekgmo', 'ekgday', 'ekgyr', 'dig', 'prop', 'nitr', 'pro', 'diuretic', 'proto', 'thaldur', 'thaltime', 'met', 'thalach', 'thalrest', 'tpeakbps', 'tpeakbpd', 'dummy', 'trestbpd', 'exang', 'xhypo', 'oldpeak', 'slope', 'rldv5', 'rldv5e', 'ca', 'restckm', 'exerckm', 'restef', 'restwm', 'exeref', 'exerwm', 'thal', 'thalsev', 'thalpul', 'earlobe', 'cmo', 'cday', 'cyr', 'num', 'lmt', 'ladprox', 'laddist', 'diag', 'cxmain', 'ramus', 'om1', 'om2', 'rcaprox', 'rcadist', 'lvx1', 'lvx2', 'lvx3', 'lvx4', 'lvf', 'cathef', 'junk', 'unk01' ,'unk02' ,'unk03' ,'unk04' ,'unk05' ,'unk06' ,'unk07' ,'unk08' ,'unk09' ,'unk10' ,'unk11' ,'unk12' ,'unk13' ,'unk14' ,'name']
    , 'file_header_78' : ['id', 'ccf', 'age', 'sex', 'painloc', 'painexer', 'relrest', 'pncaden', 'cp', 'trestbps', 'htn', 'chol', 'smoke', 'cigs', 'years', 'fbs', 'dm', 'famhist', 'restecg', 'ekgmo', 'ekgday', 'ekgyr', 'dig', 'prop', 'nitr', 'pro', 'diuretic', 'proto', 'thaldur', 'thaltime', 'met', 'thalach', 'thalrest', 'tpeakbps', 'tpeakbpd', 'dummy', 'trestbpd', 'exang', 'xhypo', 'oldpeak', 'slope', 'rldv5', 'rldv5e', 'ca', 'restckm', 'exerckm', 'restef', 'restwm', 'exeref', 'exerwm', 'thal', 'thalsev', 'thalpul', 'earlobe', 'cmo', 'cday', 'cyr', 'num', 'lmt', 'ladprox', 'laddist', 'diag', 'cxmain', 'ramus', 'om1', 'om2', 'rcaprox', 'rcadist', 'lvx1', 'lvx2', 'lvx3', 'lvx4', 'lvf', 'cathef', 'junk', 'name']
    , 'miao_header_28' : ['age', 'sex', 'cp', 'trestbps', 'htn', 'chol',        'restecg', 'ekgmo', 'ekgday', 'ekgyr', 'prop', 'nitr', 'pro', 'thaldur', 'thalach', 'thalrest', 'tpeakbps', 'tpeakbpd', 'dummy', 'trestbpd', 'exang', 'xhypo', 'oldpeak',                        'cmo', 'cday', 'cyr', 'num', 'lvf']
#   , 'uci_header_14'  : ['age', 'sex', 'cp', 'trestbps',        'chol', 'fbs', 'restecg',                                                               'thalach',                                                          'exang',          'oldpeak', 'slope', 'ca', 'thal',                       'num']
    , 'min_header_10'  : ['age', 'sex', 'cp', 'trestbps',        'chol',        'restecg',                                                               'thalach',                                                          'exang',          'oldpeak', 'num']
    # potentially of interest (append to uci when -ca = uci+)
    , 'interesting_headers'  : [
        'cigs'      # 14 cigs (cigarettes per day)
        , 'years'   # 15 years (number of years as a smoker)
        # removed "history of diabetes" because it is not collected often enough
#        , 'dm'      # 17 dm (1 = history of diabetes; 0 = no such history)
        , 'famhist' # 18 famhist: family history of coronary artery disease (1 = yes; 0 = no)
    ]
    # University of California, Irvine (most cited data set - headers)
    , 'uci_header_14'  : [
        'age'       # numeric
                        # 3 age: age in years
        , 'sex'     # binary
                        # 4 sex: sex (1 = male; 0 = female)
        , 'cp'      # ordinal
                        # 9 cp: chest pain type
                            # 1: typical angina
                            # 2: atypical angina
                            # 3: non-anginal pain
                            # 4: asymptomatic
                        # Jason's Reorder by perceived severity
                            # 0: asymptomatic
                            # 1: typical angina
                            # 2: atypical angina
                            # 3: non-anginal pain
                    # based on Detrano, 1984, p542/2
                    # "...chest pain consisted of 4952 symptomatic individuals studied with 
                    #  coronary arteriography and 23,996 asymptomatic persons who died of 
                    #  disease other than cardiac disease and were not known to have coronary
                    #  disease previous to their postmortem examinations."
        , 'trestbps'# numeric
                        # 10 trestbps: resting blood pressure (in mm Hg on admission to the hospital)
        , 'chol'    # numeric
                        # 12 chol: serum cholestoral in mg/dl
        , 'fbs'     # binary
                        # 16 fbs: (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)
        , 'restecg' # ordinal
                        # 19 restecg: resting electrocardiographic results
                            # 0: normal
                            # 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
                            # 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
        , 'thalach' # numeric
                        # 32 thalach: maximum heart rate achieved
        , 'exang'   # binary
                        # 38 exang: exercise induced angina (1 = yes; 0 = no)
        , 'oldpeak' # numeric
                        # 40 oldpeak = ST depression induced by exercise relative to rest
        , 'slope'   # ordinal
                        # 41 slope: the slope of the peak exercise ST segment
                            # 1: upsloping
                            # 2: flat
                            # 3: downsloping
        , 'ca'      # numeric
                        # 44 ca: number of major vessels (0-3) colored by flourosopy
        , 'thal'    # ordinal
                        # 51 thal:
                            # 3 = normal
                            # 6 = fixed defect
                            # 7 = reversable defect
                        # Jason's Reorder by perceived severity
                            # 0 = normal
                            # 1 = reversable defect
                            # 2 = fixed defect
        , 'num'     # numeric
                        # 58 (0-4) number of major vessels with > 50% diameter narrowing
        ]
    # attribut type by column name
    , 'col_type'  : {
        'age'       : 'numeric'
        , 'sex'     : 'binary'
        , 'cp'      : 'ordinal'
        , 'trestbps': 'numeric'
        , 'chol'    : 'numeric'
        , 'fbs'     : 'binary'
        , 'restecg' : 'ordinal'
        , 'thalach' : 'numeric'
        , 'exang'   : 'binary'
        , 'oldpeak' : 'numeric'
        , 'slope'   : 'ordinal'
        , 'ca'      : 'numeric'
        , 'thal'    : 'ordinal'
        , 'num'     : 'numeric'
        , 'cigs'    : 'numeric'
        , 'years'   : 'numeric'
        , 'famhist' : 'binary'
    }
    # Discard the following because the heart-disease.names file makes clear these columns are irrelevant for prediction purposes
    , 'useless_headers': [
        'id'        # 1 id: patient identification number
        , 'ccf'     # 2 ccf: social security number
        , 'ekgmo'   # 20 ekgmo (month of exercise ECG reading)
        , 'ekgday'  # 21 ekgday(day of exercise ECG reading)
        , 'ekgyr'   # 22 ekgyr (year of exercise ECG reading)
        , 'dummy'   # 36 dummy
        , 'cmo'     # 55 cmo: month of cardiac cath (sp?)
        , 'cday'    # 56 cday: day of cardiac cath (sp?)
        , 'cyr'     # 57 cyr: year of cardiac cath (sp?)
        , 'name'    # 76 or 91 name: last name of patient
        # Discard as not relevent for diagnosis because the describe the conditions of the ECG testing
        , 'dig'     # 23 dig (digitalis used furing exercise ECG: 1 = yes; 0 = no)
        , 'prop'    # 24 prop (Beta blocker used during exercise ECG: 1 = yes; 0 = no)
        , 'nitr'    # 25 nitr (nitrates used during exercise ECG: 1 = yes; 0 = no)
        , 'pro'     # 26 pro (calcium channel blocker used during exercise ECG: 1 = yes; 0 = no)
        , 'diuretic'# 27 diuretic (diuretic used used during exercise ECG: 1 = yes; 0 = no)
        # Discard the major vessels: attributes 59 through 68 are vessels)
        , 'lmt'     # 59 lmt
        , 'ladprox' # 60 ladprox
        , 'laddist' # 61 laddist
        , 'diag'    # 62 diag
        , 'cxmain'  # 63 cxmain
        , 'ramus'   # 64 ramus
        , 'om1'     # 65 om1
        , 'om2'     # 66 om2
        , 'rcaprox' # 67 rcaprox
        , 'rcadist' # 68 rcadist
        # Discard the following because the heart-disease.names file says they are "not used"
        , 'thalsev' # 52 thalsev: not used
        , 'thalpul' # 53 thalpul: not used
        , 'earlobe' # 54 earlobe: not used
        , 'lvx1'    # 69 lvx1: not used
        , 'lvx2'    # 70 lvx2: not used
        , 'lvx3'    # 71 lvx3: not used
        , 'lvx4'    # 72 lvx4: not used
        , 'lvf'     # 73 lvf: not used
        , 'cathef'  # 74 cathef: not used
        , 'junk'    # 75 junk: not used
        # Discard the following "unknown" headers
                    # 76-91 unk (Jason added to new.data output file new_orig.csv)
        , 'unk01' ,'unk02' ,'unk03' ,'unk04' ,'unk05' ,'unk06' ,'unk07' ,'unk08' ,'unk09' ,'unk10' ,'unk11' ,'unk12' ,'unk13' ,'unk14' 
        ]
}

# set the headers to keep based on the command line argument -cs
if args.columnset != 'max':
    if args.columnset == 'uci':
        local_dic['useful_headers'] = local_dic['uci_header_14']
    elif args.columnset == 'uci+':
        local_dic['useful_headers'] = local_dic['uci_header_14'] + local_dic['interesting_headers']
    elif args.columnset == 'min':
        local_dic['useful_headers'] = local_dic['min_header_10']

# a regular expression to match the root of the file name supplied
match = re.search(r'([a-zA-Z0-9_-]+).csv', args.filename)
#match = re.search(r'(./[a-zA-Z0-9_- ])/([a-zA-Z0-9_-]+).csv', args.filename)
# get the filename root
if match:
    if match.group(1):
        local_dic['file_root'] = match.group(1)
else:
    if args.verbosity > 0:
        print(f"Warning: no match for file name ({args.filename})")

if args.verbosity > 1:
    print(f"file root ({local_dic['file_root']})")

# determine if the longer header is required
# match = re.search(r'(new)', local_dic['file_root'])
# if match:
#     # Unreliable Header (new.dat)
#     # max_file_header_92
#     local_dic['file_header'] = ['id', 'ccf', 'age', 'sex', 'painloc', 'painexer', 'relrest', 'pncaden', 'cp', 'trestbps', 'htn', 'chol', 'smoke', 'cigs', 'years', 'fbs', 'dm', 'famhist', 'restecg', 'ekgmo', 'ekgday', 'ekgyr', 'dig', 'prop', 'nitr', 'pro', 'diuretic', 'proto', 'thaldur', 'thaltime', 'met', 'thalach', 'thalrest', 'tpeakbps', 'tpeakbpd', 'dummy', 'trestbpd', 'exang', 'xhypo', 'oldpeak', 'slope', 'rldv5', 'rldv5e', 'ca', 'restckm', 'exerckm', 'restef', 'restwm', 'exeref', 'exerwm', 'thal', 'thalsev', 'thalpul', 'earlobe', 'cmo', 'cday', 'cyr', 'num', 'lmt', 'ladprox', 'laddist', 'diag', 'cxmain', 'ramus', 'om1', 'om2', 'rcaprox', 'rcadist', 'lvx1', 'lvx2', 'lvx3', 'lvx4', 'lvf', 'cathef', 'junk', 'unk01' ,'unk02' ,'unk03' ,'unk04' ,'unk05' ,'unk06' ,'unk07' ,'unk08' ,'unk09' ,'unk10' ,'unk11' ,'unk12' ,'unk13' ,'unk14' ,'name']

# read in the header from the input file
with open(local_dic['file_dir'] + local_dic['file_root'] + '.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        local_dic['file_header'] = row
        break

my_data = np.genfromtxt(local_dic['file_dir'] + local_dic['file_root'] + '.csv', delimiter=',', skip_header=1)
# if args.verbosity > 0:
#     print(f"shape before header clean: {my_data.shape}")

missing_dic = {} # dict to hold column completeness information
# initialize missing_dic to all zeros
for col_name in local_dic['file_header']:
    missing_dic[col_name] = 0
#print(f"missing_dic: {missing_dic}")

row_count = len(my_data)
# loop through data adding up missing column counts
for row in range(0, row_count):
    for col in range(0, len(my_data[row])):
        if my_data[row][col] < 0:
            # increment found missing col value
            missing_dic[local_dic['file_header'][col]] += 1

bad_cols = 0 # just a simple count variable
col_num = 0 # the index of the columns used to add to the delete_cols variable
local_dic['clean_header_csv'] = '' # header to use when printing newly cleaned file
local_dic['clean_header'] = []  # header list to use when determining valid rows
local_dic['delete_cols'] = []   # column names to discard from data frame
# Loop through the Missing_dic to determine which columns to discard
for key, na_count in missing_dic.items():
    # reset discard boolean
    discard_header = False
    # Discard previously determined useless headers
    if key in local_dic['useless_headers']:
        discard_header = True
    # Discard previously determined useless headers
    elif args.columnset != 'max' and key not in local_dic['useful_headers']:
        discard_header = True
    elif (row_count - na_count)/row_count < args.columntolerance:
        discard_header = True
    else:
        # keep the headers that were not discarded during the above clensing
        local_dic['clean_header'].append(key) # add clean column to header
        if len(local_dic['clean_header_csv']) < 1:
            local_dic['clean_header_csv'] += key
        else:
            local_dic['clean_header_csv'] += ',' + key
        # save the index for columns of interest for ease of later processing
        if key == local_dic['target_col_name']:
            local_dic['target_col_index'] = len(local_dic['clean_header']) - 1
        # a little too "data specific" but it's quick
        elif key == 'cp':
            local_dic['cp_col_index'] = len(local_dic['clean_header']) - 1
        elif key == 'age':
            local_dic['age_col_index'] = len(local_dic['clean_header']) - 1
        elif key == 'cigs':
            local_dic['cigs_col_index'] = len(local_dic['clean_header']) - 1
        elif key == 'years':
            local_dic['years_col_index'] = len(local_dic['clean_header']) - 1
    
    # add disregarded headers to be deleted
    if discard_header is True:
        bad_cols += 1
        if args.verbosity > 1:
            print(f"{bad_cols} : {key} \t{na_count} of {row_count} = {row_count - na_count}\t{round((row_count - na_count)/row_count,2)} < {round(args.columntolerance,2)}\t{round(100*(row_count - na_count)/row_count,2)}% good")
#            print(f"{round((row_count - na_count)/row_count,2)} < {round(args.columntolerance,2)}")
        local_dic['delete_cols'].append(col_num) # add the column number for deletion

    col_num += 1 # keep track of the columns for the deletion process

# Delete the columns discarded in the above for loop
clean_data = np.delete(my_data, local_dic['delete_cols'], axis=1)
if args.verbosity > 0:
    print(f"shape before header clean: {my_data.shape}")
    print(f"shape after header clean: {clean_data.shape}")

local_dic['col_count'] = len(local_dic['clean_header'])
local_dic['delete_rows'] = []
bad_row_attributes = {}
# loop through data adding up missing row counts
for row in range(0, row_count):
    bad_row_attributes[row] = []
    for col in range(0, len(clean_data[row])):
        if clean_data[row][col] < 0:
            bad_row_attributes[row].append(local_dic['clean_header'][col])

# loop through the rows and add the bad rows to be discarded based on the passed in threshold
for row in bad_row_attributes:
    bad_attributes_count = len(bad_row_attributes[row])
    if (local_dic['col_count'] - bad_attributes_count)/local_dic['col_count'] < args.rowtolerance:
        local_dic['delete_rows'].append(row) # add columns to be discarded
        # print debug info
        if args.verbosity > 1:
            print(f"bad({bad_attributes_count}) {bad_row_attributes[row]} |\t{bad_attributes_count} of {local_dic['col_count']} = {local_dic['col_count'] - bad_attributes_count} < {round(local_dic['col_count'] - (local_dic['col_count'] * args.rowtolerance),2)}\t{round(100*(local_dic['col_count'] - bad_attributes_count)/local_dic['col_count'],2)}% good")

print(f"bad rows = {len(local_dic['delete_rows'])}")
# Delete the columns discarded in the above for loop
clean_data = np.delete(clean_data, local_dic['delete_rows'], axis=0)
clean_shape = clean_data.shape
if args.verbosity > 0:
    print(f"shape after row clean: {clean_shape}")

# sum the unique target values
unique_target_dic = {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0}
for row in range(0, clean_shape[0]):
    unique_target_dic[clean_data[row][local_dic['target_col_index']]] += 1

# recalibrate the "cp" attribut so that "4: asymptomatic" = 0 instead of 4
for row in range(0, clean_shape[0]):
    if clean_data[row][local_dic['cp_col_index']] == 4:
        clean_data[row][local_dic['cp_col_index']] = 0

# print debugging info
if args.verbosity > 0:
    for key, value in unique_target_dic.items():
        print(f"{key}:{value} = {round(100*value/clean_shape[0],2)}%")

# calculate and add "smoke" column based on ( (years smoked / age) * cigs per day )
if local_dic['cigs_col_index'] > -1:
    col_to_add = []
    local_dic['clean_header'].append('smoke')
    local_dic['clean_header_csv'] += ',smoke'
    for row in range(0, clean_shape[0]):
        smoke_val = 0
        cigar_multiplier = 1
        if clean_data[row][local_dic['years_col_index']] > 0:
            # assume less than a quarter pack of "cigs" per day means full cigars
            if clean_data[row][local_dic['cigs_col_index']] < 6:
                cigar_multiplier = 10
            # assume less than half a pack of "cigs" per day means mini cigars
            elif clean_data[row][local_dic['cigs_col_index']] < 10:
                cigar_multiplier = 5
            smoke_val = (clean_data[row][local_dic['years_col_index']] / clean_data[row][local_dic['age_col_index']]) * (clean_data[row][local_dic['cigs_col_index']] * cigar_multiplier)
        col_to_add.append( [ smoke_val ] )

    #print(f"smoke: {col_to_add}")
    #print(f"smoke: {len(col_to_add)}")
    clean_data = np.append(clean_data, col_to_add, axis=1)

    # remove 'orig" from file name and replece it with 'smoke_'
    local_dic['file_root'] = re.sub(r'orig', 'smoke_' + args.columnset, local_dic['file_root'])
else:
    # remove 'orig" from file name and replece it with 'clean'
    local_dic['file_root'] = re.sub(r'orig', 'clean_' + args.columnset, local_dic['file_root'])
    #local_dic['file_root'] = re.sub(r'orig', 'min', local_dic['file_root'])

# add binary "target" column based on "num" column: 0 = 1 & (1-4) = 1
# col_to_add = []
# local_dic['clean_header'].append('target')
# local_dic['clean_header_csv'] += ',target'
# for row in range(0, clean_shape[0]):
#     target_val = 0
#     if clean_data[row][local_dic['target_col_index']] > 0:
#         target_val = 1
#     col_to_add.append( [ target_val ] )
# clean_data = np.append(clean_data, col_to_add, axis=1)

# write new CSV file
np.savetxt(local_dic['file_dir'] + local_dic['file_root'] + '.csv', clean_data, delimiter=',', fmt='%f', comments='', header=local_dic['clean_header_csv'])


#clean_data.groupby(local_dic['target_col_index']).transform(lambda x: (x - x.mean()) / x.std())


# normal_data = normalize(clean_data, axis=0, norm='max')
# # normalized - write new CSV file
# np.savetxt(local_dic['file_dir'] + local_dic['file_root'] + '_norm.csv', normal_data, delimiter=',', fmt='%f', header=local_dic['clean_header_csv'])






















# columns potentially of interest...
    #   5 painloc: chest pain location (1 = substernal; 0 = otherwise)
    #   6 painexer (1 = provoked by exertion; 0 = otherwise)
    #   7 relrest (1 = relieved after rest; 0 = otherwise)
    #   -- or --
    #   8 pncaden (sum of 5, 6, and 7)
    #   -- and --
    #  14 cigs (cigarettes per day)
    #  15 years (number of years as a smoker)
    #  18 famhist: family history of coronary artery disease (1 = yes; 0 = no)
    #  28 proto: exercise protocol
    #       1 = Bruce     
    #       2 = Kottus
    #       3 = McHenry
    #       4 = fast Balke
    #       5 = Balke
    #       6 = Noughton 
    #       7 = bike 150 kpa min/min  (Not sure if "kpa min/min" is what was 
    #           written!)
    #       8 = bike 125 kpa min/min  
    #       9 = bike 100 kpa min/min
    #      10 = bike 75 kpa min/min
    #      11 = bike 50 kpa min/min
    #      12 = arm ergometer
