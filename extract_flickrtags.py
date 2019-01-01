import glob
import numpy as np
import natsort

def get_tags(filename):
    file_stream = open(filename,'r')
    file_list = [tag for tag in file_stream]
    file_stream.close()
    return file_list

dir_names = glob.glob('tags/*')
natsort_key = natsort.natsort_keygen()
dir_names.sort(key=natsort_key)

filenames = []

for dir_name in dir_names:
    sub_filenames = glob.glob(dir_name+'/*.txt')
    sub_filenames.sort(key=natsort_key)
    filenames += sub_filenames

hits = dict()
for filename in filenames:
    print('Parsing',filename)
    tags = get_tags(filename)
    if tags:
        for tag in tags:
            if tag in hits:
                hits[tag]+=1
            else:
                hits[tag]=1

frequent_tags=[key for key,_ in sorted(hits.items(),key=lambda kv:kv[1])[-2000:]]
frequent_tag_frequency=[val for _,val in sorted(hits.items(),key=lambda kv:kv[1])[-2000:]]



