import os.path
from os import path
import pandas
from shutil import copy2

df = pandas.read_csv('Analyze.csv')

i=0
for file in sorted(os.listdir('Auscultation_recordings')):
    if file == df["PATIENT ID"][i]:
        for filee in sorted(os.listdir(f'Auscultation_recordings/{file}')):
            name = (str(filee))
            if name[0:14] ==  df["Track Name"][i]:
                if path.exists(f"Available_Data_Rec/{file}"):
                    copy2(f"Auscultation_recordings/{file}/{filee}", f"Available_Data_Rec/{file}")
                else:
                    try:
                        os.mkdir(f"Available_Data_Rec/{file}")
                    except OSError:
                        print("Creation of the directory failed")
                    else:
                        print("Successfully created the directory ")

                    copy2(f"Auscultation_recordings/{file}/{filee}", f"Available_Data_Rec/{file}")
        i+=1


