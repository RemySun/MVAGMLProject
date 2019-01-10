import torch
import torchvision
import numpy
from PIL import Image
import glob
import torchvision.transforms as transforms
import numpy as np
import pickle
import natsort

natsort_key = natsort.natsort_keygen()

batch_size=10

data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

model = torchvision.models.alexnet(pretrained=True)
model.classifier=model.classifier[:6]
#model.fc = torch.nn.Dropout()
model.eval()
filenames = (glob.glob("mirflickr/im*.jpg"))
filenames.sort(key=natsort_key)

features = []
for batch in range(len(filenames)//batch_size):
    print(batch)
    batch_images = [data_transforms(Image.open(filename)) for filename in filenames[batch_size*batch:batch_size*(batch+1)]]
    batch_tensor = torch.stack(batch_images)
    batch_features = model(batch_tensor).data.numpy()
    features.append(batch_features)

remaining_images = [data_transforms(Image.open(filename)) for filename in filenames[batch_size*(batch+1):]]
remaining_tensor = torch.stack(remaining_images)
remaining_features = model(remaining_tensor).data.numpy()
features.append(remaining_features)

features=np.concatenate(features)
pickle.dump(features,open("FLICKR_alexnet_feat.p",'wb'))
