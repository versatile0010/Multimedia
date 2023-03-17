import numpy as np
import pyaudio
import keyboard

def convolution(signal, kernel):
    n_sig = signal.size
    n_ker = kernel.size
    rev_kernel = kernel[::-1].copy()
    result = np.zeros(n_sig - n_ker, dtype=np.int16)
    for i in range(n_sig - n_ker):
        if filter_on:
            result[i] = np.dot(signal[i:i+n_ker], rev_kernel)
        else:
            result[i] = signal[i+n_ker]
    signal[0:n_ker] = signal[n_sig-n_ker:n_sig]
    return result


RATE = 16000
CHUNK = int(RATE / 10)
kernel_size = 99
kernel = np.full(kernel_size, 1 / kernel_size)  # n(tab) = 9
in_data = np.zeros(CHUNK + kernel_size, dtype=np.int16)  # kernel_size 만큼 더해주는 이유는 previous data 저장하기 위함
filter_on = False

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                input=True, output=True, frames_per_buffer=CHUNK, input_device_index=0)

while (True):
    samples = stream.read(CHUNK)
    in_data[kernel_size:kernel_size + CHUNK] = np.frombuffer(samples, dtype=np.int16)
    out = convolution(in_data, kernel)  # real time conv
    # out = in_data
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
