import sys
import pandas as pd
from sqlalchemy import create_engine

messages_filepath = r'C:\Users\ogzpython\Desktop\ml\response_ml\Disaster_Response_Project\data\disaster_messages.csv'
categories_filepath = r'C:\Users\ogzpython\Desktop\ml\response_ml\Disaster_Response_Project\data\disaster_categories.csv'
database_filename = r'dis_res'

sys.argv.clear()
sys.argv.append('process_data.py')
sys.argv.append(messages_filepath)
sys.argv.append(categories_filepath)
sys.argv.append(database_filename)


def load_data(messages_filepath, categories_filepath):
   
    #input: file paths of csv files
    #output: creates combined data frame of two csv files
    
    #reading files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merges dataframes with key ID
    df = pd.merge(left=messages,right=categories, how='inner',on=['id'])
    
    return df

def clean_data(df):
    
    #input: combined dataframe
    #output: cleaned dataframe
    
    #there are 36 different categories in a single string
    #they can be splitted by using ";" in between 
    categories= df['categories'].str.split(expand= True, pat= ';')
    
    #since we splitted categories column names were not readable
    #data shape is like category_name-value so we can drive column names from any of the rows in the raw data
    #for that purpose we got first row
    row= categories[:1]
    #lambda function to extract name exept lats two since they are the value 
    col_names_lambda= lambda x: x[0][:-2]
    category_colnames= list(row.apply(col_names_lambda))
    #column labels asigned by pandas replaced with col names we created
    categories.columns = category_colnames
    
    
    #we have to get rid of the string attached to the actual value
    #in this loop we extract last letter in every category 
    for col in categories.columns:
        categories[col]= categories[col].apply(lambda x: x[-1])
        categories[col]= categories[col].astype(int)
        
    #due to error in data rows may hava different value than binary
    #to fix this loop again every col and convert those values to "1"
    for i in categories.columns:
        categories.loc[(categories[i]!=0) & (categories[i]!=1) ,i] = 1   
    
    #removing original categories column from the dataframe
    df.drop(labels=['categories'],axis=1,inplace=True)
    #expanding df with the categories columns we created
    df = pd.concat([df,categories],axis=1)
    #removing duplicates from the data
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):  
    
    #input: previously created df and file name for the database
    
    
    engine = create_engine('sqlite:///{}.db'.format(database_filename))
    df.to_sql('{}'.format(database_filename), engine, index=False, if_exists='replace')
    
    
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filename = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filename))
        save_data(df, database_filename)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
    
    
    

    
