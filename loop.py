import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FixedLocator
from matplotlib.image import AxesImage

# Parameters
CHUNK = 2048  # Buffer size
KEY = 8
RATE = 44100
FPS = 100  # Lower frame rate to reduce CPU usage

# Initialize array for storing audio data
audio_data = np.zeros(CHUNK)
# State of DeltaNet
np.random.seed(0)
state = np.zeros((KEY, CHUNK))
to_key = np.random.rand(CHUNK, KEY) / KEY**0.5
lr = 1

def update_audio_data(stream, chunk):
    """Update audio data from stream"""
    global audio_data
    try:
        data = np.frombuffer(stream.read(chunk, exception_on_overflow=False), dtype=np.int16)
        audio_data = data / 2**16
    except IOError as e:
        print(f"Error recording: {e}")

def forward(state, x, beta):    
    """DeltaNet

    k_t, v_t = x_t, x_t
    w_t = w_{t-1} + beta * (v_t - w_{t-1} k_t) * k_t^T
    """
    key, write = x @ to_key, x
    read = key @ state
    delta = write - read
    update = np.outer(delta, key).T
    return state + beta * update, read

def update_input(frame, iline, smat: AxesImage, oline, stream, chunk):
    """Update input and output plots with audio data from stream"""
    global state, lr
    update_audio_data(stream, chunk)
    #write = audio_data / CHUNK**0.5
    write = audio_data / (np.linalg.norm(audio_data) + 0.1)
    iline.set_ydata(write)
    state, read = forward(state, write, lr)
    smat.set_data(state)
    print(state)
    oline.set_ydata(read)
    return iline, smat, oline

# Set up audio stream
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Configure matplotlib
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 8
plt.rcParams["toolbar"] = "None"
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams['xtick.major.size'] = 1
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.size'] = 1
plt.rcParams['ytick.major.width'] = 1

# Set up input signal plot
fig, (iax, sax, oax) = plt.subplots(nrows=3, ncols=1, height_ratios=(1,1,1))
fig.canvas.manager.set_window_title('Learning')
fig.canvas.manager.resize(1500, 1000)
x = np.arange(0, 2 * CHUNK, 2)
iline, = iax.plot(x, np.random.rand(CHUNK))
iax.set_title('input')
yrange = 0.2
iax.set_ylim(-yrange, yrange)
iax.set_xlim(0, 2 * CHUNK)
iax.xaxis.set_major_locator(FixedLocator(2**np.arange(8,12)))
iax.yaxis.set_major_locator(FixedLocator(np.linspace(-yrange, yrange, 6)))

sax.set_title('state')
smat = sax.matshow(state, cmap='rainbow', vmin=-0.1, vmax=0.1, aspect='auto')
sax.xaxis.set_major_locator(FixedLocator(2**np.arange(8,12)))
sax.yaxis.set_major_locator(FixedLocator(2**np.arange(8,12)))
# remove spines
sax.spines['top'].set_visible(False)
sax.spines['right'].set_visible(False)
sax.spines['bottom'].set_visible(False)
sax.spines['left'].set_visible(False)

oax.set_title('output')
oline, = oax.plot(x, np.random.rand(CHUNK))
oax.set_xlim(0, 2 * CHUNK)
oax.set_ylim(-yrange, yrange)
oax.xaxis.set_major_locator(FixedLocator(2**np.arange(8,12)))
oax.yaxis.set_major_locator(FixedLocator(np.linspace(-yrange, yrange, 6)))

# Set up icon
#et_icon(icon_path)

# Animate plots
ani = FuncAnimation(fig, update_input, fargs=(iline, smat, oline, stream, CHUNK), interval=1000 // FPS, blit=True, save_count=CHUNK)

# Display plot
plt.show(block=True)

## Cleanup
#stream.stop_stream()
#stream.close()
#p.terminate()
