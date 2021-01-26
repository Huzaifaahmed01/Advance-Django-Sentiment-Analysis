import pandas as pd
import numpy as np

df = pd.read_csv("dataset_emotion.csv")
dfFinal = pd.DataFrame(columns=["text","react"])

df_react = df[['react_angry','react_haha','react_love','react_sad','react_wow']]
df_react = df_react.values.tolist()

dfFinal.loc[:, 'text'] = df['message'].apply(lambda x: str(x).replace('\n','. ').replace('\t',' '))

def maxReact(x):
        dataEmotion = ['angry','joyful','lovely','sorrowful','surprising']
    a = 0
    for i in x:
        if i > a:
            a = i
    return dataEmotion[x.index(a)]

listEmotions = []
for x in df_react:
    abc = maxReact(x)
    listEmotions.append(abc)

reactArray = np.array(listEmotions)


dfFinal.loc[:, 'react'] = reactArray

dfFinal.to_csv('datasetFinal.csv', index=False)

dfFinal['text'].to_csv('reveiws.txt', header=None, index=None, mode='a')
dfFinal['react'].to_csv('labels.txt', header=None, index=None, mode='a')
