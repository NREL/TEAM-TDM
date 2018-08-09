import pandas as pd
import os
import requests
import ml_battery.utils as utils

DATA_DIR = "../data/NHTS_2017"
CODEBOOK_PATH = os.path.join(DATA_DIR, "codebook.xlsx")

class Files(object):
    household_onehot = os.path.join(DATA_DIR, "household_onehot.csv")
    household = os.path.join(DATA_DIR, "hhpub.csv")
    person = os.path.join(DATA_DIR, "perpub.csv")
    trip = os.path.join(DATA_DIR, "trippub.csv")
    vehicle = os.path.join(DATA_DIR, "vehpub.csv")
    
class load_nhts(object):
    ''' Handy methods for loading in nhts 2017 dataset '''
    @staticmethod
    def _fetch():
        url = "https://nhts.ornl.gov/assets/2016/download/Csv.zip"
        if not os.path.exists(DATA_DIR):
            print("Retrieving nhts data.  Please hold.")
            zip_file = utils.download_a_thing(url, "temp.zip")
            os.makedirs(DATA_DIR)
            utils.unzip_a_thing(zip_file, DATA_DIR)
    @staticmethod         
    def _fetch_codebook():
        url = "https://nhts.ornl.gov/assets/codebook.xlsx"
        if not os.path.exists(CODEBOOK_PATH):
            print("Retrieving nhts codebook.  Please hold.")
            utils.download_a_thing(url, CODEBOOK_PATH)
    @staticmethod    
    def codebook(sheet_name=0):
        ''' load the nhts codebook... change sheet_name to, e.g. 3 to get vehicle codes '''
        load_nhts._fetch_codebook()
        return pd.read_excel(CODEBOOK_PATH, sheet_name=sheet_name)     
    @staticmethod
    def household():
        ''' load the household nhts dataset '''
        load_nhts._fetch()
        return pd.read_csv(Files.household)
    @staticmethod
    def person():
        ''' load the person nhts dataset '''
        load_nhts._fetch()
        return pd.read_csv(Files.person)
    @staticmethod
    def trip():
        ''' load the trip nhts dataset '''
        load_nhts._fetch()
        return pd.read_csv(Files.trip)
    @staticmethod
    def vehicle():
        ''' load the vehicle level nhts dataset '''
        load_nhts._fetch()
        return pd.read_csv(Files.vehicle)
        
        
        
        
        