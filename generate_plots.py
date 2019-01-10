import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval

def printMetric(result_array,name):
    metric_string=name+" & "
    for method in result_array:
        metric_string += '$'+str(round(100*method[0],1)) + '\pm' +str(round(100*method[1],1)) + '$ & '
    metric_string=metric_string[:-2]+'\\\\ \hline'
    return (metric_string)

filenames =[['log_norm_1','log_norm_2','log_norm_3'],['log_sem_1','log_sem_2','log_sem_3'],['log_struc_1','log_struc_2','log_struc_3'],['log_glob_1','log_glob_2','log_glob_3'],['log_resnet_1','log_resnet_2','log_resnet_3'],['log_walks_1','log_walks_2','log_walks_3'],['log_rand_1','log_rand_2','log_rand_3']]

recalls_img_to_txt=[]
recalls_txt_to_img=[]

maps_img_to_txt=[]
maps_txt_to_img=[]

medrs_img_to_txt=[]
medrs_txt_to_img=[]

prec_scopes_img_to_txt=[]
prec_scopes_txt_to_img=[]

for filenames_exp in filenames:
    recall_img_to_txt=[]
    recall_txt_to_img=[]

    map_img_to_txt=[]
    map_txt_to_img=[]

    medr_img_to_txt=[]
    medr_txt_to_img=[]

    prec_scope_img_to_txt=[]
    prec_scope_txt_to_img=[]
    for filename_run in filenames_exp:
        file_content = [line for line in open(filename_run,'r')]

        img_to_txt=literal_eval(file_content[-3])
        txt_to_img=literal_eval(file_content[-1])

        recall_img_to_txt.append(img_to_txt[0])
        recall_txt_to_img.append(txt_to_img[0])

        map_img_to_txt.append(img_to_txt[3])
        map_txt_to_img.append(txt_to_img[3])

        medr_img_to_txt.append(img_to_txt[1]+1)
        medr_txt_to_img.append(txt_to_img[1]+1)

        prec_scope_img_to_txt.append(img_to_txt[2])
        prec_scope_txt_to_img.append(txt_to_img[2])

    recalls_img_to_txt.append((np.mean(recall_img_to_txt,axis=0),np.std(recall_img_to_txt,axis=0)))
    recalls_txt_to_img.append((np.mean(recall_txt_to_img,axis=0),np.std(recall_txt_to_img,axis=0)))

    maps_img_to_txt.append((np.mean(map_img_to_txt,axis=0),np.std(map_img_to_txt,axis=0)))
    maps_txt_to_img.append((np.mean(map_txt_to_img,axis=0),np.std(map_txt_to_img,axis=0)))

    medrs_img_to_txt.append((np.mean(medr_img_to_txt,axis=0),np.std(medr_img_to_txt,axis=0)))
    medrs_txt_to_img.append((np.mean(medr_txt_to_img,axis=0),np.std(medr_txt_to_img,axis=0)))

    prec_scopes_img_to_txt.append((np.mean(prec_scope_img_to_txt,axis=0),np.std(prec_scope_img_to_txt,axis=0)))
    prec_scopes_txt_to_img.append((np.mean(prec_scope_txt_to_img,axis=0),np.std(prec_scope_txt_to_img,axis=0)))

recalls_img_to_txt=np.array(recalls_img_to_txt)
recalls_txt_to_img=np.array(recalls_txt_to_img)

maps_img_to_txt=np.array(maps_img_to_txt)
maps_txt_to_img=np.array(maps_txt_to_img)

medrs_img_to_txt=np.array(medrs_img_to_txt)
medrs_txt_to_img=np.array(medrs_txt_to_img)

print('Image to text')
Ks=['1','5','10']
print(printMetric(maps_img_to_txt,"MAP@all (%)"))
for i in range(3):
    print(printMetric(recalls_img_to_txt[:,:,i],"R@"+Ks[i]+" (%)"))
print(printMetric(medrs_img_to_txt/100,"Med r"))

print('Text to img')

print(printMetric(maps_txt_to_img,"MAP@all (%)"))
for i in range(3):
    print(printMetric(recalls_txt_to_img[:,:,i],"R@"+Ks[i]+" (%)"))
print(printMetric(medrs_txt_to_img/100,"Med r"))


labels=["SSPE","SSPE Sem","SSPE Struct",'SSPE Recons','SSPE Resnet','SSPE Walks','Random','DSPE']
for i in range(len(filenames)):
    plt.errorbar([200,500,1000,1500,2000,2500,3000,3500,4000,4500],prec_scopes_img_to_txt[i][0],yerr=prec_scopes_img_to_txt[i][1],label=labels[i],alpha=0.8)
plt.xlabel('Scope K')
plt.ylabel('Precision @ K')
plt.legend()
plt.tight_layout()
plt.savefig('report/img_to_txt.png')

plt.figure()
for i in range(len(filenames)):
    plt.errorbar([200,500,1000,1500,2000,2500,3000,3500,4000,4500],prec_scopes_txt_to_img[i][0],yerr=prec_scopes_txt_to_img[i][1],label=labels[i],alpha=0.8)
plt.legend()
plt.xlabel('Scope K')
plt.ylabel('Precision @ K')
plt.tight_layout()
plt.savefig('report/txt_to_img.png')
plt.show()
