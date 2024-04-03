import pandas as pd
import numpy as np

def create_new_features(X : pd.DataFrame):
    # this function will (in a sense) create new features at the stage where we have no missing values
    # quartile sum of all incoming transactions
    for idx, quarter in enumerate([0, 3, 6, 9]):
        X[f"inc_transactions_Quartile{idx+1}"] = X[f"inc_transactions_H{quarter}"] + X[f"inc_transactions_H{quarter+1}"] + X[f"inc_transactions_H{quarter+2}"]

    # quratile sum of all outcoming transactions
    for idx, quarter in enumerate([0, 3, 6, 9]):
        X[f"out_transactions_Quartile{idx+1}"] = X[f"out_transactions_H{quarter}"] + X[f"out_transactions_H{quarter+1}"] + X[f"out_transactions_H{quarter+2}"]

    #quartile sum of values of incoming transactions
    for idx, quarter in enumerate([0, 3, 6, 9]):
        X[f"inc_transactions_amt_Quartile{idx+1}"] = X[f"inc_transactions_amt_H{quarter}"] + X[f"inc_transactions_amt_H{quarter+1}"] + X[f"inc_transactions_amt_H{quarter+2}"]

    #quartile sum of values of outcoming transactions
    for idx, quarter in enumerate([0, 3, 6, 9]):
        X[f"out_transactions_amt_Quartile{idx+1}"] = X[f"out_transactions_amt_H{quarter}"] + X[f"out_transactions_amt_H{quarter+1}"] + X[f"out_transactions_amt_H{quarter+2}"]

    #quartile sum of os_term_loan
    for idx, quarter in enumerate([0, 3, 6, 9]):
        X[f"Os_term_loan_Quartile{idx+1}"] = X[f"Os_term_loan_H{quarter}"] + X[f"Os_term_loan_H{quarter+1}"] + X[f"Os_term_loan_H{quarter+2}"]

    #quartile sum of os_credit_card
    for idx, quarter in enumerate([0, 3, 6, 9]):
        X[f"Os_credit_card_Quartile{idx+1}"] = X[f"Os_credit_card_H{quarter}"] + X[f"Os_term_loan_H{quarter+1}"] + X[f"Os_term_loan_H{quarter+2}"]

    #quartile sum of os_mortgage
    for idx, quarter in enumerate([0, 3, 6, 9]):
        X[f"Os_mortgage_Quartile{idx+1}"] = X[f"Os_mortgage_H{quarter}"] + X[f"Os_mortgage_H{quarter+1}"] + X[f"Os_mortgage_H{quarter+2}"]

    #time in current job / time in address
    X["TimeInJobPerTimeInAddress"] = X["Time_in_current_job"]/X["Time_in_address"]

    #percent of incomes that go to current_acount
    for month in range(0, 13):
        X[f"incPerCurrentAccountBalance{month}"] = np.min(X[f"inc_transactions_H{month}"]/X[f"Current_amount_balance_H{month}"], 100)
        
    #percent of incomes that go to savings_account
    for month in range(0, 13):
        X[f"incPerSavingsAccountBalance{month}"] = np.min(X[f"inc_transactions_H{month}"]/X[f"Savings_amount_balance_H{month}"], 100)


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
    #features_to_drop = (X.loc[:, (np.mean(X.isna(), axis=0) > 0).values].isna()).any().index
    features_to_drop = ['Active_mortgages', 'Active_credit_card_lines',
       'External_term_loan_balance', 'External_mortgage_balance',
       'External_credit_card_balance', 'limit_in_revolving_loans_H12',
       'limit_in_revolving_loans_H11', 'limit_in_revolving_loans_H10',
       'limit_in_revolving_loans_H9', 'limit_in_revolving_loans_H8',
       'limit_in_revolving_loans_H7', 'limit_in_revolving_loans_H6',
       'limit_in_revolving_loans_H5', 'limit_in_revolving_loans_H4',
       'limit_in_revolving_loans_H3', 'limit_in_revolving_loans_H2',
       'limit_in_revolving_loans_H1', 'limit_in_revolving_loans_H0',
       'utilized_limit_in_revolving_loans_H12',
       'utilized_limit_in_revolving_loans_H11',
       'utilized_limit_in_revolving_loans_H10',
       'utilized_limit_in_revolving_loans_H9',
       'utilized_limit_in_revolving_loans_H8',
       'utilized_limit_in_revolving_loans_H7',
       'utilized_limit_in_revolving_loans_H6',
       'utilized_limit_in_revolving_loans_H5',
       'utilized_limit_in_revolving_loans_H4',
       'utilized_limit_in_revolving_loans_H3',
       'utilized_limit_in_revolving_loans_H2',
       'utilized_limit_in_revolving_loans_H1',
       'utilized_limit_in_revolving_loans_H0']
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

    create_new_features(X)
    
    return X.drop(['Target'] + types['MM-YYYY'] + types['DD-MM-YYYY'], axis=1), X['Target']