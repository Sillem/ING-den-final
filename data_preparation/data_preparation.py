import pandas as pd
import numpy as np

def transform_data(X : pd.DataFrame):
    X = X.copy()
    X.set_index(['Customer_id'], inplace=True)
    real_variables_columns = pd.read_excel('Data_dictionary.xlsx').iloc[:42, :]
    types = {k:[] for k in real_variables_columns['Type'].unique()}
    X[X == -9999] = pd.NA
    real_variables_columns
    for feature in real_variables_columns.iterrows():
        # all variables with x on the end just land with 1-12
        if feature[1]['Column name*'] == 'Customer_id': continue
        if(feature[1]['Column name*'][-1] =='x'):
            for lag in range(13):
                types[feature[1]['Type']].append((feature[1]['Column name*'][:-1]+str(lag)).replace(' ', '_'))
        else:
            types[feature[1]['Type']].append(feature[1]['Column name*'].replace(' ', '_'))
    features_to_drop = (X.loc[:, (np.mean(X.isna(), axis=0) > 0).values].isna()).any().index
    types['Created'] = []
    
    # create features that need missing values
    types['Created'].append('hasExternal_credit_card_balance')
    types['Created'].append('hasExternal_term_loan_balance')
    types['Created'].append('hasExternal_mortgage_balance')
    types['Created'].append('hasActive_credit_card_lines')
    types['Created'].append('hasActive_mortgages')

    X['hasExternal_credit_card_balance'] = ~pd.isna(X['External_credit_card_balance'])
    X['hasExternal_term_loan_balance'] = ~pd.isna(X['External_term_loan_balance'])
    X['hasExternal_mortgage_balance'] = ~pd.isna(X['External_mortgage_balance'])
    X['hasActive_credit_card_lines'] = ~pd.isna(X['Active_credit_card_lines'])
    X['hasActive_mortgages'] = ~pd.isna(X['Active_mortgages'])

    # here we drop features that are missing, at this point we have 
    X = X.drop(features_to_drop, axis=1)

    
    return X.drop(['Target'] + types['MM-YYYY'] + types['DD-MM-YYYY'], axis=1), X['Target']