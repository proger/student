# student

> Hey chat, what is the easiest way to do the real-time animation of the sound coming in from the microphone on the back? It needs to be super catchy.

A student is a loop that stores observations using the [delta learning rule](https://arxiv.org/abs/2102.11174) when the gradients are predicted using a random projection of the input. The delta rule arises from plugging in an MSE objective to train a linear model into SGD.

Playing music to the model:
[![Playing music to the model](https://img.youtube.com/vi/1t7AWa4SMlo/maxresdefault.jpg)](https://youtu.be/1t7AWa4SMlo)

Voicing vowels and playing them back from the state. The random projections are now unit norm:

[![Synthesizing speech from the model](https://img.youtube.com/vi/Tc5gUi-eNDs/maxresdefault.jpg)](https://youtu.be/Tc5gUi-eNDs)
