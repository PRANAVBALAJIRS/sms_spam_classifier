import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 

#loading the data from csv file to a pandas DataFrame
file='spam.csv'

import chardet
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

data = pd.read_csv(file,encoding='Windows-1252')

pb=pd.DataFrame(data)
pb

#drop the null columns
pb.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"], inplace=True)

#replace the null values with a null string
sms_data=pb.where((pd.notnull(pb)),'')

#Label spam as 0 and ham as 1
sms_data.loc[sms_data['v1']=='spam','v1',]=0
sms_data.loc[sms_data['v1']=='ham','v1',]=1

X = sms_data['v2']
Y = sms_data['v1']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=3)

#transform the text data into feature vectors
feature_extraction = TfidfVectorizer(min_df = 1,stop_words='english',lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

#convert the y train and y test values as integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_features, Y_train)

#prediction on training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train,prediction_on_training_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test,prediction_on_test_data)

import gradio as gr

def sms(input_sms):
  input_data_features = feature_extraction.transform([input_sms])
  prediction = model.predict(input_data_features)
  if prediction == 1:
    return "The sms received is a ham"
  else:
    return "The sms received is a spam"

interface = gr.Interface(
    fn = sms, 
    inputs = gr.inputs.Textbox(lines = 5,placeholder="Enter your input here....."), 
    outputs="text",
    examples = [['I am waiting machan. Call me once you free.'],
                ['\n FreeMsg Hey there darling its been 3 weeks now and no word back! Id like some fun you up for it still? Tb ok! XxX std chgs to send, å£1.50 to rcv'],
                ['\n Thanks for picking up the trash.'],
                ['\n Ok Im gonna head up to usf in like fifteen minutes'],
                ['\n This is the 2nd time we have tried 2 contact u. U have won the å£750 Pound prize. 2 claim is easy, call 087187272008 NOW1! Only 10p per minute. BT-national-rate.']],
    live = True,
    flagging_options = ["YES","NO","Maybe"]  
    )

interface.launch(auth=("ARNAV PRADHAN","ai&ml"),auth_message=("Check your LOGIN details(HINT: made by Pranav Balaji R S and Aravindhan B)"))