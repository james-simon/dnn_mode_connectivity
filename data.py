import os
import torch
import torchvision
import torchvision.transforms as transforms

SHUFFLE_ONE_CLASS = False

class Transforms:

    class CIFAR10:

        class VGG:

            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        class ResNet:

            train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

        class Noise:
            class NoiseTransform(object):
                def __call__(self, img):
                    x = int(img[0][0][0].item()*(10**3) + img[1][5][5].item()*(10**4) + img[2][10][10].item()*(10**5) + img[0][15][15].item()*(10**6) + img[1][20][20].item()*(10**7) + img[2][25][25].item()*(10**8))
                    # print("seed " + str(x))
                    torch.manual_seed(x)
                    torch.cuda.manual_seed(x)

                    new_img = img.clone().normal_(0, 1)
                    return new_img

            train = transforms.Compose([
                transforms.ToTensor(),
                NoiseTransform(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            test = transforms.Compose([
                transforms.ToTensor(),
                NoiseTransform(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    CIFAR100 = CIFAR10


def shuffle(x):
    x = torch.Tensor(x)
    idx = torch.randperm(x.nelement())
    x = x.reshape(-1)[idx].view(x.size())
    x = x.numpy()
    return(x)

def loaders(dataset, path, batch_size, num_workers, transform_name, use_test=False,
            shuffle_train=True):
    ds = getattr(torchvision.datasets, dataset)
    path = os.path.join(path, dataset.lower())
    transform = getattr(getattr(Transforms, dataset), transform_name)
    train_set = ds(path, train=True, download=True, transform=transform.train)

    if use_test:
        print('You are going to run models on the test set. Are you sure?')
        test_set = ds(path, train=False, download=True, transform=transform.test)
    else:
        n_train = N_TRAIN
        n_test = N_TEST
        offset = LABEL_OFFSET
        
        print("Using train (%d) + validation (%d) with label offset %d" % (n_train, n_test, offset))

        train_set.data = train_set.data[:n_train]#[:-5000]
        train_set.targets = train_set.targets[offset:(n_train + offset)]

        test_set = ds(path, train=True, download=True, transform=transform.test)
        test_set.train = False
        test_set.data = test_set.data[-n_test:]#[-5000:]
        test_set.targets = test_set.targets[(len(test_set.targets) - n_test - offset):(len(test_set.targets) - offset)]

        # if offset == 0:
        #     test_set.targets = test_set.targets[-n_test:]
        # else:
        #     test_set.targets = test_set.targets[(-n_test - offset):(-offset)]

        # train_set.train_data = train_set.train_data[:-5000]
        # train_set.train_labels = train_set.train_labels[:-5000]

        # test_set = ds(path, train=True, download=True, transform=transform.test)
        # test_set.train = False
        # test_set.test_data = test_set.train_data[-5000:]
        # test_set.test_labels = test_set.train_labels[-5000:]
        # delattr(test_set, 'train_data')
        # delattr(test_set, 'train_labels')

    if SHUFFLE_ONE_CLASS:
        for i in range(len(train_set)):
            if train_set.targets[i] == 9:
                train_set.data[i] = shuffle(train_set.data[i])
        for i in range(len(test_set)):
            if test_set.targets[i] == 9:
                test_set.data[i] = shuffle(test_set.data[i])

    return {
               'train': torch.utils.data.DataLoader(
                   train_set,
                   batch_size=batch_size,
                   shuffle=shuffle_train,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'test': torch.utils.data.DataLoader(
                   test_set,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ),
           }, max(train_set.targets) + 1
