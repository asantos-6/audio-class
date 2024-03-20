import IPython.display as ipd
from IPython.core.display import display
import librosa
import librosa.display
import numpy as np
#import madmom
import matplotlib.pyplot as plt
import torch
import torchaudio
import os
import scipy

class Audio:
    def __init__(self, file_path=None, ndarray=None, sample_rate=None):
        self.file_path = file_path
        self.ndarray = ndarray
        self.sample_rate = sample_rate
        self.num_samples = None
        self.duration_seconds = None

        if self.file_path and self.ndarray is None:
            self.load()

        self.calculateSamplesAndSeconds()        

    def __repr__(self):
        return (f"Audio(file_path='{self.file_path}', "
                f"sample_rate={self.sample_rate}, "
                f"num_samples={self.num_samples}, "
                f"duration_seconds={self.duration_seconds:.2f})")

    def play(self):
        if self.ndarray.any():
            return display(ipd.Audio(self.ndarray, rate=self.sample_rate, autoplay=True))
        else:
            raise ValueError("Audio ndarray is not loaded.")
        
    def load(self, mono=True):
        self.ndarray, self.sample_rate = librosa.load(self.file_path, mono=mono)

    def save(self, file_path=None):
        if not file_path:
            if not self.file_path:
                raise ValueError("File path must be provided.")
            file_path = self.file_path
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        scipy.io.wavfile.write(file_path, self.sample_rate, self.ndarray.astype(np.float32))

    def plot(self):
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(self.ndarray, sr=self.sample_rate)


    def plotSpectrogram(self):
        X = librosa.stft(self.ndarray)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb, sr=self.sample_rate, x_axis='time', y_axis='log')

    def calculateSamplesAndSeconds(self):
        if not self.ndarray.any():
            raise ValueError("No audio ndarray found.")
        self.num_samples = len(self.ndarray)
        self.duration_seconds = self.num_samples / self.sample_rate

    #def plotMadmomSpectrogram(self):
        #return madmom.audio.Spectrogram(self.file_path)

    def trim(self, trim_interval):
            start_time, end_time = trim_interval[0], trim_interval[1]
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            self.ndarray = self.ndarray[start_sample:end_sample]
    
    def limitVolume(self, max_volume_dBFS=-20):
        current_volume_dBFS = 10 * np.log10(np.mean(self.ndarray ** 2))

        volume_difference = current_volume_dBFS - max_volume_dBFS

        if volume_difference > 0:
            scaling_factor = 10 ** (volume_difference / 20)  # Convert dB to linear scaling factor
            self.ndarray /= scaling_factor
    
    def checkSampleRateCompatibility(self, audio, audio_var_name):
        if not isinstance(audio, Audio):
            raise ValueError('f{audio_var_name} must be an instance of Audio.')

        # Ensuring the sample rates are the same
        if self.sample_rate != audio.sample_rate:
            raise ValueError("Sample rates of the audio objects do not match.")

    def mixAudio(self, audio_to_mix, mixing_factor=1.0):
        self.checkSampleRateCompatibility(audio_to_mix, "audio_to_mix")

        self.setSameLength(audio_to_mix)

        mixed_audio = self.ndarray + audio_to_mix.ndarray * mixing_factor

        return Audio(ndarray=mixed_audio, sample_rate=self.sample_rate)
    
    def synchAudio(self, audio_to_synch_with, lag=0):
        self.checkSampleRateCompatibility(audio_to_synch_with, "audio_to_synch_with")

        if lag == 0:
            cross_correlation = scipy.signal.correlate(self.ndarray, audio_to_synch_with.ndarray, mode='full')

            lag = np.argmax(cross_correlation) - len(self.ndarray) + 1

        audio1_adjusted = self.ndarray
        audio2_adjusted = audio_to_synch_with.ndarray
        if lag > 0:
            audio1_adjusted = self.ndarray[lag:]
            audio2_adjusted = audio_to_synch_with.ndarray[:len(audio1_adjusted)]
        elif lag < 0:
            audio2_adjusted = audio_to_synch_with.ndarray[-lag:]
            audio1_adjusted = self.ndarray[:len(audio2_adjusted)]

    def setSameLength(self, audio_to_pad):
        length = max(len(self.ndarray), len(audio_to_pad.ndarray))

        self.ndarray = np.pad(self.ndarray, (0, length - len(self.ndarray)), 'constant')
        audio_to_pad.ndarray = np.pad(audio_to_pad.ndarray, (0, length - len(audio_to_pad.ndarray)), 'constant')


def rave(file_path, audio, sr, model='percussion', trim=True):
    file_name = file_path.split('/')[-1].split('.')[0]

    folder = f'../results/rave/'
    if not trim:
        folder = folder + 'full_songs/'
    folder = folder + model

    # Check if the folder exists
    if not os.path.exists(folder):
        # If it doesn't exist, create it
        os.makedirs(folder)

    output_file = f'{folder}/{file_name}-{model}.mp3'

    # Load the model
    model = torch.jit.load(f'../models/{model}.ts')
    

    #x, sr = li.load('audio.wav',sr=44100)
    #x = torch.from_numpy(x).reshape(1,1,-1)
    
    x = torch.from_numpy(audio).reshape(1,1,-1)
    #audio_tensor = torch.from_numpy(audio)
    #input_tensor = torch.unsqueeze(torch.unsqueeze(audio_tensor, 0), 0)
    #x = torch.unsqueeze(torch.unsqueeze(audio_tensor, 0), 0)

    # encode and decode the audio with RAVE
    with torch.no_grad():
        z = model.encode(x)
        x_hat = model.decode(z)
        waveform_tensor = torch.squeeze(x_hat, 0)
    #.detach().numpy().reshape(-1)


    # Apply the pre-trained model and squeeze it
    #with torch.no_grad():
     #   output_tensor = model(input_tensor)
    #waveform_tensor = torch.squeeze(output_tensor, 0)

    # extra_samples = waveform_tensor.shape[1] - audio.shape[0]
    # if extra_samples > 0:
    # Remove initial rhythms
    # waveform_tensor = waveform_tensor[:, extra_samples*36:]

    # RAVE adds extra initial samples that must be removed to synchronize with the original song.
    #offset = 24000
    #waveform_tensor = waveform_tensor[:, offset:]
    # Save the tensor into an audio file and load it
    torchaudio.save(output_file, waveform_tensor, sr)
    return output_file

def rave_mixing(audio_input, path, drums, no_drums, song, MODEL='GMDrums_v3_29-09_3M_streaming', NO_DRUMS_MIXING_FACTOR=30):
    # Assuming rave function returns a file path, this needs to be loaded into an Audio instance
    rave_path = rave(path, audio_input.ndarray, audio_input.sample_rate, model=MODEL)
    rave_output = Audio(file_path=rave_path)  # Assuming rave_output is initialized with a file_path

    print("Original song")
    song.play()

    print('Audio input (taps)')
    audio_input.play()

    print('RAVE output')
    rave_output.play()

    print('Drum track')
    drums.play()

    print('Reconstructed Drum track')
    rdrums_path = rave(path, drums.ndarray, drums.sample_rate, model=MODEL)
    rave_drums = Audio(file_path=rdrums_path)  # Load rave processed drums
    rave_drums.play()

    print('Taps + drum track')
    drums_version = drums.mixAudio(audio_input)
    drums_version.play()

    print('Taps + RAVE')
    audio_input.synchAudio(rave_output)
    rave_version = audio_input.mixAudio(rave_output)
    rave_version.play()

    print('Original + reconstructed drums')
    drums.synchAudio(rave_drums)
    drums_mix = drums.mixAudio(rave_drums, mixing_factor=5)
    drums_mix.play()

    print('Remixing the taps onto the original song')
    taps_version = no_drums.mixAudio(audio_input, mixing_factor=NO_DRUMS_MIXING_FACTOR)                  
    taps_version.play()

    print("Remixing RAVE's output onto the original song")
    rave_version = no_drums.mixAudio(rave_output, mixing_factor=NO_DRUMS_MIXING_FACTOR)
    rave_version.play()

    print("Remixing RAVE's output onto the original track with drums")
    rave_version = song.mixAudio(rave_output, mixing_factor=NO_DRUMS_MIXING_FACTOR*2)
    rave_version.play()

    print("Remixing RAVE's reconstructed drums onto the original song")
    rave_version = no_drums.mixAudio(rave_drums, mixing_factor=2)
    rave_version.play()

    return