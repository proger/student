import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FixedLocator

# Parameters
CHUNK = 2048  # Buffer size
RATE = 44100
FPS = 100  # Lower frame rate to reduce CPU usage

# Initialize array for storing audio data
audio_data = np.zeros(CHUNK)

# Function to update audio data
def update_audio_data(stream, chunk):
    global audio_data
    try:
        data = np.frombuffer(stream.read(chunk, exception_on_overflow=False), dtype=np.int16)
        audio_data = data
    except IOError as e:
        print(f"Error recording: {e}")

# Function to update input plot
def update_input(frame, iline, oline, stream, chunk):
    update_audio_data(stream, chunk)
    iline.set_ydata(audio_data / 2**16)
    oline.set_ydata(audio_data / 2**16)
    return iline, oline

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
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 4
plt.rcParams['ytick.major.size'] = 1
plt.rcParams['ytick.major.width'] = 1

# Set up input signal plot
fig, (iax, oax) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, height_ratios=(3,1))
fig.canvas.manager.set_window_title('Student')
x = np.arange(0, 2 * CHUNK, 2)
iline, = iax.plot(x, np.random.rand(CHUNK))
iax.set_title('input')
yrange = 0.5
iax.set_ylim(-yrange, yrange)
iax.set_xlim(0, 2 * CHUNK)
iax.xaxis.set_major_locator(FixedLocator(2**np.arange(8,12)))
iax.yaxis.set_major_locator(FixedLocator(np.linspace(-yrange, yrange, 6)))
oax.set_title('output')
oline, = oax.plot(x, np.random.rand(CHUNK))

# Set up icon
#et_icon(icon_path)

# Animate plots
ani = FuncAnimation(fig, update_input, fargs=(iline, oline, stream, CHUNK), interval=1000 // FPS, blit=True, save_count=CHUNK)

# Display plot
plt.show()

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()
