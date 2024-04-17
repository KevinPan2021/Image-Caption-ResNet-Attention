import matplotlib.pyplot as plt


# visualize examples
def visualize_samples(element):
    fig, axs = plt.subplots(2, 1, figsize=(6,4))
    
    axs[0].set_title(element.get_name())
    img = element.get_image().numpy().transpose(1,2,0)
    axs[0].imshow(img)
    axs[0].axis('off')

    if isinstance(element.get_caption(), list):
        axs[1].text(0.5, 0.5, '\n'.join(element.get_caption()), fontsize=10, va='center', ha='center', wrap=True)
    elif isinstance(element.get_caption(), str):
        axs[1].text(0.5, 0.5, element.get_caption(), fontsize=10, va='center', ha='center', wrap=True)
    axs[1].axis('off')
    
   

def display_results(img, ground_truth, pred):

    fig, axs = plt.subplots(3, 1, figsize=(6,4))
    
    img = img.numpy().transpose(1,2,0)
    axs[0].imshow(img)
    axs[0].axis('off')


    axs[1].text(0.5, 0.5, ground_truth, fontsize=10, va='center', ha='center', wrap=True)
    axs[1].axis('off')
    
    axs[2].text(0.5, 0.5, pred, fontsize=10, va='center', ha='center', wrap=True)
    axs[2].axis('off')
    plt.tight_layout()


# plot the loss and acc curves
def plot_training_curves(train_BLUE, train_loss, valid_BLUE, valid_loss):
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.title('loss curves')
    plt.xlabel('epochs')
    plt.ylabel('cross entropy loss')
    plt.legend()
    
    plt.figure()
    plt.plot(train_BLUE, label='train')
    plt.plot(valid_BLUE, label='valid')
    plt.title('BLUE curves')
    plt.xlabel('epochs')
    plt.ylabel('BLUE score')
    plt.legend()
    
    
    
    