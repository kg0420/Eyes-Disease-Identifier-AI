import os

base_path =r"C:\Users\Acer\OneDrive\Documents\Datasets\Eyes_Dataset\Train" # change this

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    
    if os.path.isdir(folder_path):
        count = 0
        for file in os.listdir(folder_path):
            if file.lower().endswith(image_extensions):
                count += 1
                
        print(f"{folder}: {count} images")
