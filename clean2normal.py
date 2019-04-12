########################################################################
# Jason Baumbach
#   Read in clean csv file, normalize contents, then output new csv file
########################################################################

# required for easy dataframe manipulations
import pandas as pd
# allow command line options
import argparse
# required for determining data file name on the fly
import re

parser = argparse.ArgumentParser(description="perform the k-means clustering on 1 to 2-dimensional data")
parser.add_argument("-f", "--filename", default="./data/new_smoke_uci+.csv", help="file name (and path if not in . dir)")
#parser.add_argument("-ct", "--columntolerance", type=restricted_float, default=0.0, help="tolerance percentage as a decimal for missing column data")
#parser.add_argument("-rt", "--rowtolerance", type=restricted_float, default=0.0, help="tolerance percentage as a decimal for missing row data")
#parser.add_argument("-cs", "--columnset", choices=['uci', 'uci+', 'min', 'max'], default='max', help="limit the attributes to be considered")
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=2, help="increase output verbosity")
args = parser.parse_args()

# parse input file name to use when creating the output file name
local_dic = {
    'file_root' : ''
    , 'file_dir' : './data/'
    , 'target_col_name' : 'num'
    , 'target_col_index' : -1
}

# a regular expression to match the root of the file name supplied
match = re.search(r'([a-zA-Z0-9_-]+\+?).csv', args.filename)
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

df = pd.read_csv(local_dic['file_dir'] + local_dic['file_root'] + '.csv')
print(f"df.shape:{df.shape}")

# min-max normalization by target_col_name (i.e., "num")
#df_normal = df.groupby(local_dic['target_col_name']).transform(lambda x: (x-min(x))/(max(x)-min(x)))
df_normal = df.groupby(local_dic['target_col_name']).transform(lambda x: (x-x.min())/(x.max()-x.min()))

# normalization by z-score normalization
#df_normal = df.groupby(local_dic['target_col_index']).transform(lambda x: (x - x.mean()) / x.std())

# recombine the groupby column into the dataframe to be saved
df_normal = pd.concat([df_normal,df.loc[:, local_dic['target_col_name'] ] ], axis=1)

# add the binary "target" column used in nearly all of the literature referencing this data set.
#df_normal['target'] = df_normal[local_dic['target_col_name']].apply(lambda x: 'True' if x >= 1 else 'Fals')
df_normal['target'] = df_normal[local_dic['target_col_name']].apply(lambda x: 1 if x >= 1 else 0)
print(f"{df_normal.head()}")

df_normal.to_csv(local_dic['file_dir'] + local_dic['file_root'] + '_normal.csv', index=False)