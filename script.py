import json
from flask import Flask, request
from serve import get_keys
import numpy as np
import pandas as pd
from joblib import dump, load
#from serve import get_keywords_api
# I've commented out the last import because it won't work in kernels, 
# but you should uncomment it when we build our app tomorrow

# create an instance of Flask
app = Flask(__name__)

# load our pre-trained model & function
keys = get_keys()

# Define a post method for our API.
@app.route('/extractpackages', methods=['GET','POST'])
def extractpackages():
    """ 
    Takes in a json file, extracts the keywords &
    their indices and then returns them as a json file.
    """
    # the data the user input, in json format
    model_columns = load('model_columns.joblib')
    query_df = pd.DataFrame(request.json)
    query_df=query_df.drop(['phone number','total day minutes','total eve minutes','total night minutes','total intl minutes' ],axis=1)
    query_df=query_df.drop(['churn'],axis=1)
    stateq=pd.get_dummies(query_df['state'],drop_first=True)
    intplanq=pd.get_dummies(query_df['international plan'],drop_first=True)
    vmplanq=pd.get_dummies(query_df['voice mail plan'],drop_first=True)
    intplanq = intplanq.rename(columns={'yes':'intplan'})
    vmplanq = vmplanq.rename(columns={'yes':'vmplan'})
    query_df=query_df.drop(['state','international plan','voice mail plan'],axis=1)
    query_df=pd.concat([query_df,stateq,intplanq,vmplanq],axis=1)
    query_df = query_df.reindex(columns=model_columns, fill_value=0)
    # use our API function to get the keywords
    output_data = keys(query_df)
    # convert our dictionary into a .json file
    # (returning a dictionary wouldn't be very
    # helpful for someone querying our API from
    # java; JSON is more flexible/portable)
    response = output_data
    return response
