import os
import librosa
import librosa.display
from pydub import AudioSegment
import sklearn
import wave
import contextlib
import numpy as np
import matplotlib.pyplot as plt

def printSignal(fileName):
    spf = wave.open(fileName, "r")
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, "Int32")
    fs = spf.getframerate()
    Time = np.linspace(0, len(signal) / fs, num=len(signal))
    plt.figure(1)
    plt.title("Signal Wave")
    plt.plot(Time, signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.savefig("signalWave.png")
    plt.show()

def printHistogram(directory):
    histo = []
    bins = []

    for i in range(20, 33):
        bins.append(i)
    for filename in sorted(os.listdir(directory)):
        i = 0
        if filename != '.DS_Store':
            for file in sorted(os.listdir(directory + '/' + filename)):
                fname = 'Auscultation_recordings/' + filename + '/' + file
                with contextlib.closing(wave.open(fname, 'r')) as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration = frames / float(rate)

                i += 1
                if (i==1):
                    histo.append(duration)
                    break

    plt.hist(histo, bins=bins)
    plt.title("Histogram")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')
    plt.savefig("histogram.png")
    plt.show()


def split64ms(fname):
    t1 = 0  # Works in milliseconds
    t2 = 64
    # newAudio = AudioSegment.from_wav("Auscultation_recordings/PV18212/PV18212_310119_L.wav")
    newAudio = AudioSegment.from_wav(fname)
    newAudio = newAudio[t1:t2]
    return newAudio


def split32ms(fname):
    t1 = 0  # Works in milliseconds
    t2 = 32
    # newAudio = AudioSegment.from_wav("Auscultation_recordings/PV18212/PV18212_310119_L.wav")
    newAudio = AudioSegment.from_wav(fname)
    newAudio = newAudio[t1:t2]
    return newAudio
    # array16ms[0].export('newSong.wav', format="wav")  # Exports to a wav file in the current path.


def split16ms(fname):
    t1 = 0  # Works in milliseconds
    t2 = 16
    # newAudio = AudioSegment.from_wav("Auscultation_recordings/PV18212/PV18212_310119_L.wav")
    newAudio = AudioSegment.from_wav(fname)
    newAudio = newAudio[t1:t2]

    return newAudio

def exportSplitFrames(array16, array32, array64):
    for i in range(len(array16)):
        array16[i].export(f'array16/file{i}.wav', format='wav')

    for i in range(len(array32)):
        array32[i].export(f'array32/file{i}.wav', format='wav')

    for i in range(len(array64)):
        array64[i].export(f'array64/file{i}.wav', format='wav')

def createSplitFrames(directory):
    array16 = []
    array32 = []
    array64 = []
    for filename in sorted(os.listdir(directory)):
        i = 0
        if filename != '.DS_Store':
            for file in sorted(os.listdir(directory + '/' + filename)):
                fname = 'Auscultation_recordings/' + filename + '/' + file
                array16.append(split16ms(fname))
                array32.append(split32ms(fname))
                array64.append(split64ms(fname))

                i += 1
                if (i==2):
                    break

    exportSplitFrames(array16, array32, array64)
    ZCR('array32/file10.wav')
    SpectralCentroid('Auscultation_recordings/PV18212/PV18212_310119_L.wav')
    SpectralCentroid('array32/file17.wav')
    MFCC('array32/file13.wav')

def ZCR(filename):
    x, sr = librosa.load(filename)
    plt.figure(figsize=(8, 5))
    librosa.display.waveplot(x, sr=sr)
    plt.title('Audio Signal')
    plt.savefig('images/signal.png')
    plt.figure(figsize=(14, 8))
    plt.plot(x)
    plt.grid()
    plt.savefig('images/ZCR.png')
    plt.show()
    zero_crossings = librosa.zero_crossings(x, pad=False)
    zcr = librosa.feature.zero_crossing_rate(x)
    print(sum(zero_crossings))
    print(zcr)

def SpectralCentroid(filename):
    x, sr = librosa.load(filename)
    spectral_centroids = librosa.feature.spectral_rolloff(x, sr=sr)[0]
    # Computing the time variable for visualization
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)

    # Computing the time variable for visualization
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)

    # Normalising the spectral centroid for visualisation
    def normalize(x, axis=0):
        return sklearn.preprocessing.minmax_scale(x, axis=axis)

    # Plotting the Spectral Centroid along the waveform
    librosa.display.waveplot(x, sr=sr, alpha=0.4)
    plt.plot(t, normalize(spectral_centroids), color='r')
    plt.savefig('images/SCFull.png')
    plt.show()

def MFCC(filename):
    x, sr = librosa.load(filename)
    mfccs = librosa.feature.mfcc(x, sr=sr)
    print(mfccs.shape)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.savefig('images/MFCC32.png')
    plt.show()


def main():
    printSignal("Auscultation_recordings/PV18222/PV18222_061218_L.wav")
    printHistogram('Auscultation_recordings')
    createSplitFrames('Auscultation_recordings')
    ZCR("Auscultation_recordings/PV18222/PV18222_061218_L.wav")
    playsound('Auscultation_recordings/PV18212/PV18212_310119_L.wav')

if __name__ == '__main__':
    main()
