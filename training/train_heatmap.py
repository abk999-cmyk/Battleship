import pickle, numpy as np, tensorflow as tf, tqdm, os
from tqdm.keras import TqdmCallback
from tensorflow.keras import layers, models, optimizers, losses

DATA = "data/battleship_supervised.pkl"
# The dataset file is a stream of many pickled (state, label) samples.
X_list, y_list = [], []
with open(DATA, "rb") as f:
    while True:
        try:
            state, label = pickle.load(f)       # state: 10×10×3, label: 10×10
            X_list.append(state)
            y_list.append(label[..., None])     # add channel dim
        except EOFError:
            break

X = np.array(X_list, dtype=np.float16)         # (N,10,10,3)
y = np.array(y_list, dtype=np.int8)            # (N,10,10,1)
split = int(0.9*len(X))
X_train, y_train, X_val, y_val = X[:split],y[:split],X[split:],y[split:]

model = models.Sequential([
    layers.Input(shape=(10,10,3)),
    layers.Conv2D(64,3,activation='relu',padding='same'),
    layers.Conv2D(64,3,activation='relu',padding='same'),
    layers.Conv2D(64,3,activation='relu',padding='same'),
    layers.Conv2D(1 ,1,activation='sigmoid',padding='same')
])
model.compile(optimizers.Adam(1e-3),
              loss=losses.BinaryCrossentropy(),
              metrics=['accuracy'])
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=12,
    batch_size=256,
    callbacks=[TqdmCallback(verbose=1)],
    verbose=0   # Disable Keras’ own bar so only tqdm shows
)
os.makedirs("models",exist_ok=True)
model.save("models/battleship_heatmap.h5")