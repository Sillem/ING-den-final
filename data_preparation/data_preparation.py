import pandas as pd


def transform_data(X : pd.DataFrame):
    real_variables_columns = pd.read_excel('Data_dictionary.xlsx').iloc[:42, :]
    types = {k:[] for k in real_variables_columns['Type'].unique()}

    real_variables_columns
    for feature in real_variables_columns.iterrows():
        # all variables with x on the end just land with 1-12
        if(feature[1]['Column name*'][-1] =='x'):
            for lag in range(13):
                types[feature[1]['Type']].append((feature[1]['Column name*'][:-1]+str(lag)).replace(' ', '_'))
        else:
            types[feature[1]['Type']].append(feature[1]['Column name*'].replace(' ', '_'))

    # at this time we include only numeric features
    return X[types['Integer'] + types['Float'] + types['Integer (0-330)'] + types['Integer (0 or 1)']]