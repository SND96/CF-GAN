import os
import shutil

src = '../../../raghavsidata/Data/street2shop/images2/'
src_files = os.listdir(src)

f_list = open('../../../raghavsidata/Data/street2shop/train_street_4cat_dress.txt', 'r')

dst = 'datasets/dress/images/'

# for file_name in src_files:
#     full_file_name = os.path.join(src, file_name)
#     os.system("mkdir "+dst+str(file_name[0:-4]))
#     if (os.path.isfile(full_file_name)):
#         shutil.copy(full_file_name, dst+str(file_name[0:-4]))
for line in f_list:
	name = line.split(' ')[0].split('/')[3]
	# print(name)
	full_file_name = os.path.join(src, name)
	os.system("mkdir "+dst+str(name[0:-5]))
	if (os.path.isfile(full_file_name)):
		shutil.copy(full_file_name, dst+str(name[0:-5]))	
