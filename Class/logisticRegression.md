# Logistic Regresion

## Approximation

### Linear regresion

<img src="https://latex.codecogs.com/gif.latex?f%28w%2Cb%29%20%3D%20wx%20+%20b">

### Sigmoid

<img src="https://latex.codecogs.com/gif.latex?%5Chat%7By%7D%3Dh_%7B%5Ctheta%20%7D%28x%29%3D%5Cfrac%7B1%7D%7B1+e%5E%7B-wx+b%7D%7D">

## Cost function

<img src="https://latex.codecogs.com/gif.latex?J%28w%2Cb%29%3DJ%28%5Ctheta%20%29%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bn%7D%5E%7Bi%3D1%7D%5By%5E%7Bi%7Dlog%28h_%7B%5Ctheta%20%7D%28x%5E%7Bi%7D%29%29+%281-y%5E%7Bi%7D%29log%281-h_%7B%5Ctheta%20%7D%28x%5E%7Bi%7D%29%29%5D">

## Update Rules

### Update weights

<img src="https://latex.codecogs.com/gif.latex?w%20%3D%20w%20-%20a%5Ccdot%20dw">

### Update bias

<img src="https://latex.codecogs.com/gif.latex?b%20%3D%20b%20-%20a%5Ccdot%20dw">
