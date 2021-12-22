import csv, pandas
import librosa
import numpy as np
import os

header = 'filename Patient_ID spectral_centroid spectral_rolloff zero_crossing_rate'
for i in range(1, 6):
    header += f' mfcc{i}'
header = header.split()

df = pandas.read_csv('Analyze.csv')

file = open('newData1.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
i=0
j=0
for filee in sorted(os.listdir('Auscultation_recordings')):
    if filee != '.DS_Store' and filee == df["PATIENT ID"][j]:
        for filename in sorted(os.listdir('Auscultation_recordings/' + filee)):
                name = (str(filename))
                if name[0:14] == df["Track Name"][j]:
                    i+=1
                    songname = f'./Auscultation_recordings/{filee}/{filename}'
                    y, sr = librosa.load(songname, mono=True, duration=30)
                    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                    print(sr)
                    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                    zcr = librosa.feature.zero_crossing_rate(y)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr)
                    to_append = f'{filename} {filee} {np.mean(spec_cent)} {np.mean(rolloff)} {np.mean(zcr)}'
                    temp = 0
                    for e in mfcc:
                        temp += 1
                        to_append += f' {np.mean(e)}'
                        if temp == 5:
                            break
                    file = open('newData1.csv', 'a', newline='')
                    with file:
                        writer = csv.writer(file)
                        writer.writerow(to_append.split())
                    if i == 2:
                        i=0
                        break
        j+=1
