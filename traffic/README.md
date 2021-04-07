# TRAFFIC 
### STEP 1: Working on gtsrb-small

The initial Network is simply a copy of the example given in the lecture. Just finding a starting point to experiment with.

1) Had to understand flattening / Global Pooling to get the output layer to match the categories

2) Center-cropping non-square images yielded worse results than directly resizing to IMG_WIDTH and IMG_HEIGHT

3) After a lot of trial an error, the variation of the parameters filters and units, resulted in stagnating low accuracy for each epoch. The variation of the activation functions did the trick. Now the accuracy improved with every epoch. 

### STEP 2: Working on gtsrb

Two steps of Convolution and Pooling resulted in good accuracy. The neural network performed even better with a Flatten layer (instead of a Global2DPooling). The next insight was, that with the correct activation functions the performance could be increased even further.

- sigmoid worked best for the hidden dense layer
- softmax worked best for categorization layer

### STEP 3: Removing complexity
The first working configuration had quite good accuracy with round about 430k params. I tried to reduce the "params" and see if less complexity would yield equal accuracy. The following table shows a few runs that have been made:

|Run|Params|Accuracy|
|---|---|---|
|1| 0.97 | 430k
|2| 0.97 | 142k
|3| 0.96 |  97k
|4| 0.96 |  65k  
|5| 0.92 |  49k

I would go for run 4. The according configuration is shown below:
```
shp = IMG_WIDTH, IMG_HEIGHT, 3

model = tf.keras.Sequential([
    Conv2D(filters=20, kernel_size=(3, 3), input_shape=shp, activation="tanh"),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=16, kernel_size=(3, 3), input_shape=shp, activation="tanh"),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(units=100, activation="sigmoid"),
    Dropout(rate=0.5),
    Dense(units=43, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```


### FINAL THOUGHTS
- The whole process was mostly trial and error.
- A guideline on how to decide the correct set of parameters would be super helpful.
  - Number of filters/units
  - Selection of activation functions
- How can I evaluate the quality/robustness of the neural network?
- My next step would be to create a parametric study, in which several parameter combinations will be tested with respect to accuracy.
