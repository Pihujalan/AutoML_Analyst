import pandas as pd

def load_dataset(source_type, path_or_url, sheet_name=None):
     try:
          
        if(source_type == 'csv'):
          df = pd.read_csv(path_or_url) 
          df.to_csv("input.csv",index=False)
          print("CSV successfully loaded and saved!")
          return df
     
        elif source_type == 'gsheet':
          import gspread
          from oauth2client.service_account import ServiceAccountCredentials
          scope = [
               "https://spreadsheets.google.com/feeds",
               "https://www.googleapis.com/auth/drive"
          ]
          creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json",scope)
          client = gspread.authorize(creds)
          sheet = client.open_by_url(path_or_url)
          worksheet= sheet.worksheet(sheet_name)
          data = worksheet.get_all_records()
          df = pd.DataFrame(data)
          df.to_csv("input.csv",index=False)
          print("Google Sheet successfully loaded and saved!")
          return df
        else:
           raise ValueError("source_type must be either 'csv' or 'gsheet' ")  

     except Exception as e:
        error_msg=str(e).split(":"[-1].strip())
        print("Error loading dataset:", error_msg)
        return None
          
if __name__ == "__main__":
     # df=load_dataset('csv','Employee_dataset.csv')
     # print(df.head())

     #df = load_dataset(
    #source_type='gsheet',
    #path_or_url='https://docs.google.com/spreadsheets/d/1dfO5hJJ8tMfExlX0T0IAUE5pjccdxhMbVWMRrbdeMKY',
    #sheet_name='Employee_dataset'
#)
    df = load_dataset(
    source_type='gsheet',
    path_or_url='https://docs.google.com/spreadsheets/d/this_is_wrong',
    sheet_name='Employee_dataset'
)
    if df is not None:
       print (df.head())

    else:
       print("Skipping dataframe operations since loading failed.")
       


    



