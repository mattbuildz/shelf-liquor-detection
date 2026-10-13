training_imgs = open("splits/train_images.txt").read().split("\n")
validation_imgs = open("splits/val_images.txt").read().split("\n")
print(len(set(training_imgs) & set(validation_imgs)))   # tyle zdjęć jest w obu listach