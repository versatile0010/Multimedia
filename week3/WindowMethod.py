import numpy as np
import pyaudio
import keyboard
import scipy.signal as signal

RATE = 16000
CHUNK = int(RATE / 10)
kernel_size = 99
kernel = np.full(kernel_size, 1 / kernel_size)  # n(tab) = 9
in_data = np.zeros(CHUNK + kernel_size, dtype=np.int16)  # kernel_size 만큼 더해주는 이유는 previous data 저장하기 위함
filter_on = False

tabs = 777
cutoff_frequency = 1000.0
fs = 16000.0
fn = fs/2
f_normal = cutoff_frequency/fn

rect = signal.firwin(tabs, cutoff=f_normal, window="boxcar")
hamming = signal.firwin(tabs, cutoff=f_normal, window="hamming")

filter = rect

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                input=True, output=True, frames_per_buffer=CHUNK, input_device_index=0)

while (True):
    samples = stream.read(CHUNK)
    audio = np.frombuffer(samples, dtype=np.int16)

    if filter_on:
        filtered = signal.lfilter(filter, 1, audio)
        out = filtered.astype(np.int16)
    else:
        out = audio.astype(np.int16)
    y = out.tobytes()
    stream.write(y)

    if keyboard.is_pressed('q'):
        break
    if keyboard.is_pressed('f'):
        if filter_on:
            filter_on = False
            print("Filter Off.")
        else:
            filter_on = True
            print("Filter On.")

# terminate
stream.stop_stream()
stream.close()
p.terminate()
