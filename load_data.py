
'''## **2. Data Loader'''
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
def load_data(DATA_PATH, num_workers, batch_size):
    data_path = DATA_PATH
    transform_dict = {
        'model': transforms.Compose(
                                    [transforms.Resize(299),
                                     transforms.CenterCrop(299),
                                     transforms.ToTensor(),
                                     ])}
    train_data = datasets.ImageFolder(root=data_path + '/train', transform=transform_dict['model'])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    
    val_data = datasets.ImageFolder(root=data_path + '/val', transform=transform_dict['model'])
    # #val_data = datasets.ImageFolder(root=data_path.replace("Saliency","Data") + '/val', transform=transform_dict['model'])
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
   
    test_data = datasets.ImageFolder(root=data_path + '/test', transform=transform_dict['model'])
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_data, train_loader, val_data, val_loader, test_data, test_loader