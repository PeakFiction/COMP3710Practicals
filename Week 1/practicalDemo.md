Demonstration 1.3

- Change the Gaussian function into a 2D sine or cosine function (1 Mark)
From Gaussian:
```
# 1.2 Conversion to PyTorch
# ----- Device -----
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ----- Params -----
sigma = 1.0
size = 100

# ----- Grid (tensors) -----
x = torch.linspace(-4, 4, size, device=device)
y = torch.linspace(-4, 4, size, device=device)
Y, X = torch.meshgrid(y, x, indexing="ij")  # shape [size, size]

# ----- 2D Gaussian -----
Z = torch.exp(-(X**2 + Y**2) / (2 * sigma**2))

# ----- Plot (move to CPU -> NumPy) -----
plt.imshow(Z.detach().cpu().numpy(), extent=(-4, 4, -4, 4), origin='lower', cmap='viridis')
plt.colorbar(label='Amplitude')
plt.title('2D Gaussian Function (PyTorch)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
`

to Sine Function
`#1.2 Gaussian to Sine Function
# ----- Device -----
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ----- Params -----
size = 100
kx, ky, phase = 4.0, 2.0, 0.0  # controls stripe frequency & orientation

# ----- Grid (tensors) -----
x = torch.linspace(-4, 4, size, device=device)
y = torch.linspace(-4, 4, size, device=device)
Y, X = torch.meshgrid(y, x, indexing="ij")  # shape [size, size]

# ----- 2D Sine -----
Z = torch.sin(kx * X + ky * Y + phase)

# ----- Plot -----
plt.imshow(Z.detach().cpu().numpy(), extent=(-4, 4, -4, 4), origin='lower', cmap='gray')
plt.colorbar(label='Amplitude')
plt.title('2D Sine Function (PyTorch)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

- What do you get when you multiply both the Gaussian and the sine/cosine function together and visualise it? (1 Mark)
Modulation of the sinusoidal pattern by a Gaussian envelope, which gives you a Gabor filter.
```
#Combine Sine Function and Gaussian
import torch
import matplotlib.pyplot as plt

# ----- Device -----
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ----- Parameters -----
size = 200             # grid resolution
sigma = 1.0            # Gaussian spread
kx, ky = 4.0, 2.0      # sine wave vector components
phase = 0.0            # sine phase shift

# ----- Grid -----
x = torch.linspace(-4, 4, size, device=device)
y = torch.linspace(-4, 4, size, device=device)
Y, X = torch.meshgrid(y, x, indexing="ij")

# ----- Gaussian -----
G = torch.exp(-(X**2 + Y**2) / (2 * sigma**2))

# ----- Sine -----
S = torch.sin(kx * X + ky * Y + phase)

# ----- Modulation (Gabor) -----
Gabor = G * S

# ----- Plot -----
plt.imshow(Gabor.detach().cpu().numpy(),
           extent=(-4, 4, -4, 4),
           origin='lower',
           cmap='gray')
plt.colorbar(label='Amplitude')
plt.title('Gabor Filter (Gaussian × Sine)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```


Demonstration 2.3
To demonstrate this part of the lab, you must show your lab demonstrator the following items:
- High resolution computation of the set by decreasing the mgrid spacing and zooming to another part of the Mandelbrot set and compute the image for it. This may increase the computation time significantly, so choose a value that balances quality of the image and time spent. (1 Mark)

Zoomed in ver.
```
#2.3 Demonstration
#Higher Resolution, Zoomed In
# Use NumPy to create a 2D array of complex numbers on [-2,2]x[-2,2]

Y, X = np.mgrid[-0.2:0.2:0.0005, -0.8:-0.7:0.0005]

#load into PyTorch sensors
x = torch.Tensor(X)
y = torch.Tensor(Y)
z = torch.complex(x, y) #important!
zs = z.clone() #Updated!
ns = torch.zeros_like(z)

#transfer to GPU device
z = z.to(device)
zs = zs.to(device)
ns = ns.to(device)

#Mandelbrot Set
for i in range(200):
    #Compute the new values of z: z^2 + x
    zs_ = zs*zs + z
    
    #Have we diverged with this new value?
    not_diverged = torch.abs(zs_) < 4.0
    
    #Update Variables to Compute
    ns = ns + not_diverged
    zs = zs_

#Plot
fig = plt.figure(figsize=(16,10))

def processFractal(a):
    """Display an array of iteration counts as a
        colorful picture of a fractal."""
    
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic),
    30+50*np.sin(a_cyclic),
    155-80*np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a
plt.imshow(processFractal(ns.cpu().numpy()))
plt.tight_layout(pad=0)
plt.show()
```

- Modify the code so to show a Julia set rather than the Mandelbrot set. (1 Mark)
```
# 2.3 Demonstration
# Julia Set
# Use NumPy to create a 2D array of complex numbers
Y, X = np.mgrid[-1.3:1.3:0.001, -2:1:0.001]

# Load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)

# Julia set: start z as grid, constant c fixed
z = torch.complex(x, y)  
c = torch.complex(torch.tensor(-0.7), torch.tensor(0.27015))  # common Julia constant
ns = torch.zeros_like(z)

# Transfer to GPU device
z = z.to(device)
c = c.to(device)
ns = ns.to(device)

# Julia Set computation
for i in range(200):
    z = z * z + c
    not_diverged = torch.abs(z) < 4.0
    ns = ns + not_diverged

# Plot
fig = plt.figure(figsize=(16,10))

def processFractal(a):
    """Display an array of iteration counts as a colorful picture of a fractal."""
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic),
                          30+50*np.sin(a_cyclic),
                          155-80*np.cos(a_cyclic)], 2)
    img[a == a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a

plt.imshow(processFractal(ns.cpu().numpy()))
plt.tight_layout(pad=0)
plt.show()
```

Demonstration 3.4
Made Triflake Fractal:
```
#Triflake Fractal

import matplotlib.pyplot as plt
import numpy as np

def koch_snowflake(order=0):
    """
    Returns x, y coordinates of Koch snowflake of given order.
    """
    # Initial equilateral triangle
    angles = np.array([0, 120, 240]) * np.pi / 180.0
    x = np.cos(angles)
    y = np.sin(angles)
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    for _ in range(order):
        new_x = []
        new_y = []
        for i in range(len(x)-1):
            # Start and end points of the segment
            x0, y0 = x[i], y[i]
            x1, y1 = x[i+1], y[i+1]
            
            # Points dividing the segment into thirds
            dx = (x1 - x0) / 3
            dy = (y1 - y0) / 3
            x_a = x0 + dx
            y_a = y0 + dy
            x_b = x0 + 2*dx
            y_b = y0 + 2*dy

            # Peak of the equilateral triangle
            angle = np.arctan2(y_b - y_a, x_b - x_a) - np.pi/3
            dist = np.sqrt(dx**2 + dy**2)
            x_peak = x_a + np.cos(angle) * dist
            y_peak = y_a + np.sin(angle) * dist

            # Append points
            new_x += [x0, x_a, x_peak, x_b]
            new_y += [y0, y_a, y_peak, y_b]
        new_x.append(x[-1])
        new_y.append(y[-1])
        x, y = np.array(new_x), np.array(new_y)

    return x, y

# --- Generate Triflake ---
order = 4  # higher = more detail
x, y = koch_snowflake(order)

# --- Plot ---
plt.figure(figsize=(8, 8))
plt.plot(x, y, color='blue')
plt.axis('equal')
plt.axis('off')
plt.title(f'Triflake (Koch Snowflake) - Order {order}')
plt.show()
```

You will need to demonstrate our fractal project by:
- showing the resulting fractal code and its output to the demonstrator and justifying that it uses PyTorch/TF in a major component within the algorithm of the fractal utilising parallelism with PyTorch/TF in a reasonable way (3 Marks)

- showing that the fractal code is available on a GitHub repository on your account. You may be asked to verify that you own the GitHub account. (1 Mark)