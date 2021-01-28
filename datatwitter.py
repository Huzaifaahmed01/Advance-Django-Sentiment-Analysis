import pandas as pd
import numpy as np

df = pd.read_csv("text_emotion.csv" )
dfFinal = pd.DataFrame(columns=["text","react"])

dfFinal.loc[:, 'text'] = df['content'].apply(lambda x: str(x).replace('\n','. ').replace('\t',' '))

def maxReact(x):
    dataEmotion = {0:'negative',2:'neutral',4:'positive'}
    return dataEmotion[x]

dfFinal.loc[:, 'react'] = df['sentiment']

print(df['sentiment'].unique())

dfFinal.to_csv('datasetTwitter.csv', index=False)
#
# dfPos = dfFinal[dfFinal['react']=='positive'].iloc[:15000]
# dfNeg = dfFinal[dfFinal['react']=='negative'].iloc[:15000]
# dfNet = dfFinal[dfFinal['react']=='neutral'].iloc[:15000]

# frames = [dfPos, dfNeg]
#
# dfFinal = pd.concat(frames)

dfFinal['text'].to_csv('Emotion_reviews.txt', header=None, index=None)
dfFinal['react'].to_csv('Emotion_labels.txt', header=None, index=None)
