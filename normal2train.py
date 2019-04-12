########################################################################
# Jason Baumbach
#   Read in normalized csv file, split contents into train & test sets, 
#   then output each into new csv file
########################################################################

# required for easy dataframe manipulations
import pandas as pd
# allow command line options
import argparse
# required for determining data file name on the fly
import re

# code adopted from:
#   https://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-between-0-0-1-0-usin
def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

parser = argparse.ArgumentParser(description="perform the k-means clustering on 1 to 2-dimensional data")
parser.add_argument("-f", "--filename", default="./data/new_smoke_uci+_normal.csv", help="file name (and path if not in . dir)")
parser.add_argument("-tp", "--trainpercent", type=restricted_float, default=0.8, help="training percentage as a decimal (1-tp=test)")
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
    , 'num_bias' : 0.9
    , 'random_state' : 100
}

# a regular expression to match the root of the file name supplied
match = re.search(r'([a-zA-Z0-9_-]+\+?[a-zA-Z0-9_-]+).csv', args.filename)
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
df_shape = df.shape
print(f"df.shape:\n{df_shape}")

# sum the unique target values
unique_target_dic = {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0}
#for row in range(0, clean_shape[0]):
for index, row in df.iterrows():
    unique_target_dic[row[local_dic['target_col_name']]] += 1

# print debugging info
if args.verbosity > 0:
    for key, value in unique_target_dic.items():
        print(f"{key}:{value} = {round(100*value/df_shape[0],2)}%")

dict_of_nums = dict(tuple(df.groupby(local_dic['target_col_name'])))

num_0 = pd.DataFrame()
num_1 = pd.DataFrame()
num_2 = pd.DataFrame()
num_3 = pd.DataFrame()
num_4 = pd.DataFrame()

for i in range(0, len(dict_of_nums)):
    if i == 0:
        num_0 = dict_of_nums[i]
    elif i == 1:
        num_1 = dict_of_nums[i]
    elif i == 2:
        num_2 = dict_of_nums[i]
    elif i == 3:
        num_3 = dict_of_nums[i]
    elif i == 4:
        num_4 = dict_of_nums[i]

# split the num=4 90/10
df_train_4 = num_4.sample(frac=local_dic['num_bias'], random_state=local_dic['random_state'])
df_test_4 = num_4.drop(df_train_4.index)
# split the num=3 90/10
df_train_3 = num_3.sample(frac=local_dic['num_bias'], random_state=local_dic['random_state'])
df_test_3 = num_3.drop(df_train_3.index)
# split the num=2 by the passed-in percentage
df_train_2 = num_2.sample(frac=args.trainpercent, random_state=local_dic['random_state'])
df_test_2 = num_2.drop(df_train_2.index)
# split the num=1 by the passed-in percentage
df_train_1 = num_1.sample(frac=args.trainpercent, random_state=local_dic['random_state'])
df_test_1 = num_1.drop(df_train_1.index)
# split the num=0 by the passed-in percentage
df_train_0 = num_0.sample(frac=args.trainpercent, random_state=local_dic['random_state'])
df_test_0 = num_0.drop(df_train_0.index)

# combine the different dataframes into training and testing data frames
df_train = pd.concat([df_train_0, df_train_1, df_train_2, df_train_3, df_train_4], axis=0, ignore_index=True)
df_test = pd.concat([df_test_0, df_test_1, df_test_2, df_test_3, df_test_4], axis=0, ignore_index=True)

print(f"df_train:{df_train.shape} | df_test:{df_test.shape}")

# print(f"df_train:\n{df_train.head()}")
# print(f"df_test:\n{df_test.head()}")

df_train.to_csv(local_dic['file_dir'] + local_dic['file_root'] + '_train.csv', index=False)
df_test.to_csv(local_dic['file_dir'] + local_dic['file_root'] + '_test.csv', index=False)





