import json
from flask import Flask, request
from serve import get_keys
import numpy as np
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
    input_data = request.args
    Y = np.array([float(input_data['feature1']), float(input_data['feature2']), float(input_data['feature3']), float(input_data['feature4'])])

    # use our API function to get the keywords
    output_data = keys(Y)
    # convert our dictionary into a .json file
    # (returning a dictionary wouldn't be very
    # helpful for someone querying our API from
    # java; JSON is more flexible/portable)
    response = output_data
    return response
