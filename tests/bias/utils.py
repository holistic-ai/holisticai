import pandas as pd
from io import StringIO

def load_bias_classification_data():
    csv_data=""",group_a,group_b,y_pred,y_true
    Item 1,1,0,1,1
    Item 2,1,0,1,1
    Item 3,1,0,1,0
    Item 4,1,0,0,0
    Item 5,0,1,1,1
    Item 6,0,1,1,1
    Item 7,0,1,0,1
    Item 8,0,1,0,0
    Item 9,0,1,0,0
    Item 10,0,1,0,0
    """
    return pd.read_csv(StringIO(csv_data))

def load_bias_regression_data():
    csv_data=""",group_a,group_b,y_pred,y_true
    Item 1,1,0,0.9,0.8
    Item 2,1,0,0.7,0.9
    Item 3,1,0,0.3,0.2
    Item 4,1,0,0.2,0.1
    Item 5,0,1,0.8,0.7
    Item 6,0,1,1,0.9
    Item 7,0,1,0.8,0.9
    Item 8,0,1,0.2,0.3
    Item 9,0,1,0.1,0.2
    Item 10,0,1,0.2,0.1
    """
    return pd.read_csv(StringIO(csv_data))

def load_bias_recommender_data():
    csv_data="""User,group_a,group_b,item_1,item_2,item_3,item_4,item_1_true,item_2_true,item_3_true,item_4_true
    User 1,1,0,0.9,0.8,0.4,0.2,0.7,0.8,0.4,0.2
    User 2,1,0,0.7,0.9,0.1,0.7,0.9,0.9,0.1,0.2
    User 3,1,0,0.3,0.2,0.3,0.3,0.3,0.8,0.2,0.6
    User 4,1,0,0.2,0.1,0.7,0.8,0.2,0.1,0.7,0.8
    User 5,0,1,0.8,0.7,0.9,0.1,0.6,0.7,0.9,0.1
    User 6,0,1,1,0.9,0.3,0.6,1,0.9,0.3,0.6
    User 7,0,1,0.8,0.9,0.1,0.1,0.8,0.1,0.1,0.1
    User 8,0,1,0.2,0.3,0.1,0.5,0.2,0.3,0.1,0.5
    User 9,0,1,0.1,0.2,0.7,0.7,0.1,0.2,0.7,0.7
    User 10,0,1,0.2,0.7,0.1,0.2,0.2,0.1,0.1,0.8
    """
    return pd.read_csv(StringIO(csv_data))