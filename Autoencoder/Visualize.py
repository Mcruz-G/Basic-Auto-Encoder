import matplotlib.pyplot as plt

def view(originalImg, encodedImg, predictedImg):
    plt.figure(figsize=(40, 4))
    for i in range(10):
        # display original images
        ax = plt.subplot(3, 20, i + 1)
        plt.imshow(originalImg[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display encoded images
        ax = plt.subplot(3, 20, i + 1 + 20)
        plt.imshow(encodedImg[i].reshape(8,4))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
    # display reconstructed images
        ax = plt.subplot(3, 20, 2*20 +i+ 1)
        plt.imshow(predictedImg[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.show()
