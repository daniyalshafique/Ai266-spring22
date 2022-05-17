#Assign-2 By DANIYAL SHAFIQUE 61428 & ASSADULLAH

CODE

import pandas as pd
import numpy as np
train_td = pd.read_csv('/content/sample_data/test.csv');
train_td
rId = train_td[['id']];
rId.insert(1,"target",0);
rId['target'] = np.random.rand(700000,1);
print(rId);
rId.to_csv('out.csv',index=False);

#KAGGLE SUBMISSION OUTPUT

![assign-2 (kaggle submission)](https://user-images.githubusercontent.com/43805740/168909956-67433c6e-64f6-4f48-9920-a3ed9564a3fc.PNG)
