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
    # Example usage:
    # df = load_dataset('csv', 'Employee_dataset.csv')
    # print(df.head())
    
    # For Google Sheets:
    # df = load_dataset(
    #     source_type='gsheet',
    #     path_or_url='YOUR_GOOGLE_SHEET_URL',
    #     sheet_name='Sheet1'
    # )
    pass
       


    



