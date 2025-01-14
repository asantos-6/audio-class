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
import wave
import io
from madmom.audio.spectrogram import LogarithmicFilteredSpectrogram

class Audio:
    def __init__(self, file_path=None, ndarray=None, sample_rate=None, stereo=False):
        self.file_path = file_path
        self.ndarray = ndarray
        self.sample_rate = sample_rate
        self.sr = sample_rate
        self.num_samples = None
        self.duration_seconds = None
        self.stereo = stereo

        if self.file_path and self.ndarray is None:
            self.load()

        self.calculate_samples_and_seconds()        

    def __repr__(self):
        return (f"Audio(file_path='{self.file_path}', "
                f"sample_rate={self.sample_rate}, "
                f"sr={self.sr}, "
                f"num_samples={self.num_samples}, "
                f"duration_seconds={self.duration_seconds:.2f})")

    def play(self):
        if self.ndarray.any():
            if not self.stereo:
                return display(ipd.Audio(self.ndarray, rate=self.sample_rate, autoplay=True))
            else:
                normalized_ndarray = np.int16(self.ndarray * (32767 / np.max(np.abs(self.ndarray))))

                # Create a byte stream
                byte_stream = io.BytesIO()
                
                # Create a WAV file using wave module
                with wave.open(byte_stream, 'wb') as wav_file:
                    wav_file.setnchannels(2)  # Stereo
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(normalized_ndarray.tobytes())

                # Set the stream position to the beginning
                byte_stream.seek(0)

                # Display the audio using ipd.Audio
                return ipd.display(ipd.Audio(byte_stream.read(), rate=self.sample_rate/2, autoplay=True))
        else:
            raise ValueError("Audio ndarray is not loaded.")
        
    @classmethod
    def play_ndarray(cls, ndarray, sample_rate=22050):
        if ndarray.any():
            return ipd.display(ipd.Audio(ndarray, rate=sample_rate, autoplay=True))
        else:
            raise ValueError("Audio ndarray is not loaded.")
        
    def load(self, mono=True):

        self.ndarray, self.sample_rate = librosa.load(self.file_path, mono=mono, sr=self.sample_rate)

    def save(self, file_path=None):
        from scipy.io import wavfile

        if file_path:  # If a file path is provided, use the original behavior
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            wavfile.write(file_path, self.sample_rate, self.ndarray.astype(np.float64))
        else:
            # Use a BytesIO buffer instead of a file
            buffer = io.BytesIO()
            wavfile.write(buffer, self.sample_rate, self.ndarray.astype(np.float64))
            buffer.seek(0)  # Move to the start of the buffer for reading if needed
            
            # You can return or use the buffer for further processing
            return buffer
        return

    def save_ndarray(self, file_path=None):
        if not file_path:
            if not self.file_path:
                raise ValueError("File path must be provided.")
            file_path = self.file_path
            
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, self.ndarray)

    def plot(self):
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(self.ndarray, sr=self.sample_rate)

    def spectrogram(self, **kwargs):
        return LogarithmicFilteredSpectrogram(self.file_path, **kwargs)

    def plot_spectrogram(self, title=None, time_range=None):
        """
        Plots the spectrogram of self.ndarray.
        
        :param title: (Optional) title for the plot.
        :param time_range: (Optional) tuple of (start_sec, end_sec).
                        If provided, the spectrogram will be plotted 
                        only for this time range in seconds.
        """
        # If time_range is provided, slice self.ndarray accordingly
        if time_range is not None:
            start_sec, end_sec = time_range
            start_sample = int(start_sec * self.sample_rate)
            end_sample = int(end_sec * self.sample_rate)
            end_sample = min(end_sample, len(self.ndarray))  # Safety check
            audio_data = self.ndarray[start_sample:end_sample]
        else:
            audio_data = self.ndarray

        # Compute STFT on the sliced array (or the full array if no range given)
        X = librosa.stft(audio_data)
        Xdb = librosa.amplitude_to_db(abs(X))

        # Plot
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb, sr=self.sample_rate, x_axis='time', y_axis='log')
        
        if title:
            plt.title(title)
        plt.show()

    def calculate_samples_and_seconds(self):
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

            self.calculate_samples_and_seconds()  

    def to_tensor(self):
        tensor = torch.from_numpy(self.ndarray)
        # return torchaudio.transforms.Resample(self.sample_rate, 16000)(tensor)
        return tensor
    
    def normalize(self):
        max_amplitude = np.max(np.abs(self.ndarray))  # Find the maximum peak
        if max_amplitude == 0:  # Prevent division by zero
            return self.ndarray  # Return unchanged if audio is silent
        normalization_factor = 1.0 / max_amplitude  # Calculate normalization factor
        self.ndarray = self.ndarray * normalization_factor  # Apply normalization
    
    def apply_compression(self, threshold=0.8, ratio=4):
        # Apply simple compression to an audio signal
        for i in range(len(self.ndarray)):
            if abs(self.ndarray[i]) > threshold:
                # Apply compression above the threshold
                self.ndarray[i] = np.sign(self.ndarray[i]) * (threshold + (abs(self.ndarray[i]) - threshold) / ratio)

    def apply_fadeout(self):
        # convert to audio indices (samples)
        end = len(self.ndarray)
        length = end // 2
        start = end - length

        # compute fade out curve
        # linear fade
        fade_curve = np.linspace(1.0, 0.0, length)

        # apply the curve
        self.ndarray[start:end] = self.ndarray[start:end] * fade_curve
        
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

        #ndarray1, ndarray2 = self.synchAudio(audio_to_mix)
        ndarray1 = self.ndarray
        ndarray2 = audio_to_mix.ndarray

        mixed_audio_left = ndarray1
        mixed_audio_right = ndarray2 * mixing_factor


        mixed_audio_stereo = np.vstack((mixed_audio_left, mixed_audio_right)).T

        return Audio(ndarray=mixed_audio_stereo, sample_rate=self.sample_rate, stereo=True)
    
    def synchAudio(self, audio_to_synch_with, lag=0):
        self.checkSampleRateCompatibility(audio_to_synch_with, "audio_to_synch_with")

        o_env1 = librosa.onset.onset_strength(y=self.ndarray, sr=self.sample_rate)
        o_env2 = librosa.onset.onset_strength(y=audio_to_synch_with.ndarray, sr=audio_to_synch_with.sample_rate)

        times = librosa.times_like(o_env1, sr=self.sample_rate)

        onset_frames1 = librosa.onset.onset_detect(onset_envelope=o_env1, sr=self.sample_rate)
        onset_frames2 = librosa.onset.onset_detect(onset_envelope=o_env2, sr=audio_to_synch_with.sample_rate)

        onset_binary1 = [1 if t in times[onset_frames1] else 0 for t in times]
        onset_binary2 = [1 if t in times[onset_frames2] else 0 for t in times]

        if lag == 0:
            cross_correlation = scipy.signal.correlate(onset_binary1, onset_binary2, mode='full')

            lag = int((np.argmax(cross_correlation) - len(self.ndarray) - 1))
        else:
            lag = int(- lag * self.sample_rate)

        print(lag)
        audio1_adjusted = self.ndarray
        audio2_adjusted = audio_to_synch_with.ndarray

        if lag > 0:
            ndarray1 = self.ndarray[lag:]
            ndarray2 = audio_to_synch_with.ndarray[:len(audio1_adjusted)]
        elif lag < 0:
            ndarray1 = audio_to_synch_with.ndarray[-lag:]
            ndarray2 = self.ndarray[:len(audio2_adjusted)]

        return ndarray1, ndarray2

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

def pad_shortest_audio(audio1, audio2):
    if audio1.num_samples < audio2.num_samples:
        audio1.ndarray = np.pad(audio1.ndarray, (0, audio2.num_samples - audio1.num_samples), 'constant')
        audio1.calculate_samples_and_seconds()
    else:
        audio2.ndarray = np.pad(audio2.ndarray, (0, audio1.num_samples - audio2.num_samples), 'constant')
        audio2.calculate_samples_and_seconds() 
