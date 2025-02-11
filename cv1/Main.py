import numpy as np
import matplotlib.pyplot as plt

arr_1d = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

arr_2d = np.array([[1,2,3],[3,2,1]])

arr_3d = np.array([[1,2,3],[3,2,1],[4,5,6]])

zeros = np.zeros((3,3))

ones = np.ones((2,4))

rng = np. arange(0,10,2)

linsp = np.linspace(0,1,5)

print('arr1d:\n',arr_1d)
print('arr2d:\n',arr_2d)
print('arr3d:\n',arr_3d)
print('\nmetody np')
print('zeros:\n',zeros)
print('ones:\n',ones)
print('rng:\n',rng)
print('linsp:\n',linsp)
print('\nvlastnosti pole')
print('arr2d shape:\n',arr_2d.shape)
print('arr2d dtype:\n',arr_2d.dtype)
print('arr2d ndim:\n',arr_2d.ndim)

# indexace od 0
print('\nindexy')
print(arr_1d[0])
print(arr_1d[1:3]) #od indexu 1 do 3
print(arr_1d[3:]) # od indexu 3 do konce

print('\narray do matice a zpět do pole')
arr = np.arange(12)
print(arr)
mat = arr.reshape((2,6))
print(mat)
flat = mat.ravel()
print(flat)

print('\nspojování a stakování pole')
a = np.array([[1,2],[3,4]])
b = np.array([[4,5],[7,8]])
print('a:\n', a)
print('b:\n',b)
c_concat = np.concatenate((a,b))
print('concat:\n',c_concat)
c_hstack = np.hstack((a,b))
print('hystack:\n',c_hstack)

x = np.linspace(0,2*np.pi,100)
y = np.sin(x)
plt.plot(x,y,label='sin(x)')
plt.title('Základní sinusovka')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.show()

x = np.random.randn(100)
y = np.random.randn(100)
plt.scatter(x,y, c="red",alpha=0.5, label="náhodné body")
plt.title("scatter plot ukázka")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

data = np.random.randn(500)
plt.hist(data, bins=20, edgecolor="black",alpha=0.7)
plt.title("Histogram rozložení")
plt.xlabel("Hodnota")
plt.ylabel("Frekvence")
plt.show()


mat_data = np.random.rand(10,10)
plt.imshow(mat_data, cmap="viridis")
plt.colorbar(label="Hodnota")
plt.title("Heatmap 10x10")
plt.show()

tensor_3d = np.random.rand(5,10,10)
fig, axes = plt.subplots(1,5,figsize=(15,3))
for i in range(5):
    axes[i].imshow(tensor_3d[i,:,:], camp="viridis")
    axes[i].set_title(f"Vrstva{i}")
    axes[i].axis("off")
plt.show()

mean_image = tensor_3d.mean(axis=0)
plt.imshow(mean_image, cmap="viridis")
plt.title("Průměr 3d dat přes 1. dimenzi")
plt.colorbar()
plt.show()
