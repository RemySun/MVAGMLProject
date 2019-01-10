import glob
import numpy as np
import natsort
import pickle

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

frequent_tags=np.sort([key for key,_ in sorted(hits.items(),key=lambda kv:kv[1])[-2000:]])
frequent_tag_frequency=[val for _,val in sorted(hits.items(),key=lambda kv:kv[1])[-2000:]]

tag_dict = dict(zip(frequent_tags,[i for i in range(2000)]))

train_tags = []
test_tags = []

for i,filename in enumerate(filenames[:25000]):
    file_tags= np.zeros(2000,dtype=np.float32)
    file_taglist = get_tags(filename)
    for tag in file_taglist:
        if tag in tag_dict:
            file_tags[tag_dict[tag]]+=1.
    if (i%5)<3:
        train_tags.append(file_tags)
    else:
        test_tags.append(file_tags)

train_tags=np.array(train_tags)
test_tags=np.array(test_tags)

pickle.dump(train_tags,open('FLICKR_train_tags.p','wb'))
pickle.dump(test_tags,open('FLICKR_test_tags.p','wb'))
