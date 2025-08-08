import os
from PIL import Image
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.utils.data as data
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode, CenterCrop, RandomHorizontalFlip, ColorJitter, RandomResizedCrop
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm
import sklearn.metrics as metrics
import logging
import pandas as pd
import ast
from clip.clip import tokenize

def _convert_to_rgb(image):
    return image.convert('RGB')

class PASCAL_VOC_2007(data.Dataset):
    def __init__(self, datapath, train=True, resolution=224):
        self.use_augment = True if train else False
        self.filelist = []
        if train:
            for _, _, files in os.walk('{}/train/resized224/VOCdevkit/VOC2007/JPEGImages'.format(datapath)):
                for filename in files:
                    if filename.endswith('.jpg'):
                        self.filelist.append('{}/train/resized224/VOCdevkit/VOC2007/JPEGImages/{}'.format(datapath, filename))
            self.df = pd.read_csv('{}/train/trainval.csv'.format(datapath))
        else:
            for _, _, files in os.walk('{}/test/resized224/VOCdevkit/VOC2007/JPEGImages'.format(datapath)):
                for filename in files:
                    if filename.endswith('.jpg'):
                        self.filelist.append('{}/test/resized224/VOCdevkit/VOC2007/JPEGImages/{}'.format(datapath, filename))
            self.df = pd.read_csv('{}/test/test.csv'.format(datapath))
        self.transform = self._build_transform(resolution)

    def _build_transform(self, resolution):
        if self.use_augment:
            transform = Compose([
                Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                RandomResizedCrop(resolution, scale=(0.8, 1.0), ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC),
                RandomHorizontalFlip(0.5),
                ColorJitter(0.1),
                _convert_to_rgb,
                ToTensor(),
                Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            transform = Compose([
                Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                CenterCrop((resolution, resolution)),
                _convert_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        return transform

    def labeltext(self):
        return ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
             "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        image_path = self.filelist[idx]
        image_name = image_path.split('/')[-1]

        image = self.transform(Image.open(image_path))
        label = torch.tensor(ast.literal_eval(self.df.loc[self.df['filename'] == image_name, 'label'].iloc[0])).float() # 20 cls

        return self.filelist[idx], image, label


class CIFAR_10(CIFAR10):
    def __init__(self, datapath, train=True, resolution=224):
        if train:
            transform = Compose([
                Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                RandomResizedCrop(resolution, scale=(0.8, 1.0), ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC),
                RandomHorizontalFlip(0.5),
                ColorJitter(0.1),
                _convert_to_rgb,
                ToTensor(),
                Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            transform = Compose([
                Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                CenterCrop((resolution, resolution)),
                _convert_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        super().__init__(root=datapath, train=train, transform=transform)

    def labeltext(self):
        return ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        out_target = torch.tensor([0] * 10).float()
        out_target[target] = 1
        return "nothing", img, out_target   # CIFAR没有文件名


class CIFAR_100(CIFAR100):
    def __init__(self, datapath, train=True, resolution=224):
        if train:
            transform = Compose([
                Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                RandomResizedCrop(resolution, scale=(0.8, 1.0), ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC),
                RandomHorizontalFlip(0.5),
                ColorJitter(0.1),
                _convert_to_rgb,
                ToTensor(),
                Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            transform = Compose([
                Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                CenterCrop((resolution, resolution)),
                _convert_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        super().__init__(root=datapath, train=train, transform=transform)

    def labeltext(self):
        return ["beaver", "dolphin", "otter", "seal", "whale", "aquarium fish", "flatfish", "ray", "shark", "trout",
                "orchids", "poppies", "roses", "sunflowers", "tulips", "bottles", "bowls", "cans", "cups", "plates",
                "apples", "mushrooms", "oranges", "pears", "sweet peppers", "clock", "computer keyboard", "lamp",
                "telephone", "television", "bed", "chair", "couch", "table", "wardrobe", "bee", "beetle", "butterfly",
                "caterpillar", "cockroach", "bear", "leopard", "lion", "tiger", "wolf", "bridge", "castle", "house",
                "road", "skyscraper", "cloud", "forest", "mountain", "plain", "sea", "camel", "cattle", "chimpanzee",
                "elephant", "kangaroo", "fox", "porcupine", "possum", "raccoon", "skunk", "crab", "lobster", "snail",
                "spider", "worm", "baby", "boy", "girl", "man", "woman", "crocodile", "dinosaur", "lizard", "snake",
                "turtle", "hamster", "mouse", "rabbit", "shrew", "squirrel", "maple", "oak", "palm", "pine", "willow",
                "bicycle", "bus", "motorcycle", "pickup truck", "train", "lawn-mower", "rocket", "streetcar", "tank", "tractor"]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        out_target = torch.tensor([0] * 100).float()
        out_target[target] = 1
        return "nothing", img, out_target   # CIFAR没有文件名


class SUN_397(data.Dataset):
    def __init__(self, datapath, train=True, resolution=224):
        self.idx_dict = {}
        self.classes = []

        if train:
            for txt_idx in ["01"]:
                txtpath = "{}/Training_{}.txt".format(datapath, txt_idx)
                with open(txtpath, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        imgname = line.strip()
                        cls = imgname.split("/")[2]
                        if cls not in self.classes:
                            self.classes.append(cls)
                        category = self.classes.index(cls)
                        self.idx_dict[imgname] = {
                            "class": cls,
                            "category": category
                        }
            self.transform = Compose([
                Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                RandomResizedCrop(resolution, scale=(0.8, 1.0), ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC),
                RandomHorizontalFlip(0.5),
                ColorJitter(0.1),
                _convert_to_rgb,
                ToTensor(),
                Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            for txt_idx in ["01"]:
                txtpath = "{}/Testing_{}.txt".format(datapath, txt_idx)
                with open(txtpath, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        imgname = line.strip()
                        cls = imgname.split("/")[2]
                        if cls not in self.classes:
                            self.classes.append(cls)
                        category = self.classes.index(cls)
                        self.idx_dict[imgname] = {
                            "class": cls,
                            "category": category
                        }
            self.transform = Compose([
                Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                CenterCrop((resolution, resolution)),
                _convert_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

        self.resized_datapath = datapath + "/resized224"
        self.imgnames = list(self.idx_dict.keys())

    def __len__(self):
        return len(self.idx_dict)

    def labeltext(self):
        return self.classes

    def __getitem__(self, idx):
        imgname = self.imgnames[idx]
        image_path = "{}/SUN397/{}".format(self.resized_datapath, imgname)
        image = self.transform(Image.open(image_path))

        label = torch.tensor([0] * 362).float() # 362 cls
        label[self.idx_dict[imgname]["category"]] = 1

        return imgname, image, label


class Oxford_IIIT_Pets(data.Dataset):
    def __init__(self, datapath, train=True, resolution=224):
        self.idx_dict = {}
        self.classes = []

        if train:
            self.resized_datapath = datapath + "/train/images"
            self.transform = Compose([
                Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                RandomResizedCrop(resolution, scale=(0.8, 1.0), ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC),
                RandomHorizontalFlip(0.5),
                ColorJitter(0.1),
                _convert_to_rgb,
                ToTensor(),
                Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            self.resized_datapath = datapath + "/test/images"
            self.transform = Compose([
                Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                CenterCrop((resolution, resolution)),
                _convert_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

        for _, _, files in os.walk(self.resized_datapath):
            for imgname in files:
                if imgname.endswith(".jpg"):
                    cls = ' '.join(imgname.split('_')[:-1])
                    if cls not in self.classes:
                        self.classes.append(cls)
                    category = self.classes.index(cls)
                    self.idx_dict[imgname] = {
                        "class": cls,
                        "category": category
                    }
        self.imgnames = list(self.idx_dict.keys())

    def __len__(self):
        return len(self.idx_dict)

    def labeltext(self):
        return self.classes

    def __getitem__(self, idx):
        imgname = self.imgnames[idx]
        image_path = "{}/{}".format(self.resized_datapath, imgname)
        image = self.transform(Image.open(image_path))

        label = torch.tensor([0] * 37).float() # 37 cls
        label[self.idx_dict[imgname]["category"]] = 1

        return imgname, image, label


class Caltech_101(data.Dataset):
    def __init__(self, datapath, train=True, resolution=224):
        self.idx_dict = {}
        self.classes = []

        if train:
            self.resized_datapath = datapath + "/train/101_ObjectCategories"
            self.transform = Compose([
                Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                RandomResizedCrop(resolution, scale=(0.8, 1.0), ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC),
                RandomHorizontalFlip(0.5),
                ColorJitter(0.1),
                _convert_to_rgb,
                ToTensor(),
                Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            self.resized_datapath = datapath + "/test/101_ObjectCategories"
            self.transform = Compose([
                Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                CenterCrop((resolution, resolution)),
                _convert_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        for root, _, files in os.walk(self.resized_datapath):
            for imgname in files:
                cls = os.path.basename(root)
                if cls != "BACKGROUND_Google" and imgname.endswith(".jpg"):
                    if cls not in self.classes:
                        self.classes.append(cls)
                    category = self.classes.index(cls)
                    self.idx_dict[cls + '/' + imgname] = {
                        "class": cls,
                        "category": category
                    }
        self.imgnames = list(self.idx_dict.keys())

    def __len__(self):
        return len(self.idx_dict)

    def labeltext(self):
        return self.classes

    def __getitem__(self, idx):
        imgname = self.imgnames[idx]
        image_path = "{}/{}".format(self.resized_datapath, imgname)
        image = self.transform(Image.open(image_path))

        label = torch.tensor([0] * 101).float() # 101 cls
        label[self.idx_dict[imgname]["category"]] = 1

        return imgname, image, label


def downstreamtest(model, downstream_dataset, downstream_task, epoch_main, device, onlyclip=False):

    BATCH_SIZE = 32
    NUM_WORKERS = 2

    if downstream_dataset == "PASCAL":
        num_classes = 20
        train_dataset = PASCAL_VOC_2007("/path/to/your/PASCAL_VOC_2007", True, 224)
        val_dataset = PASCAL_VOC_2007("path/to/your/PASCAL_VOC_2007", False, 224)

    elif downstream_dataset == "CIFAR10":
        num_classes = 10
        train_dataset = CIFAR_10("/path/to/your/CIFAR_10", True, 224)
        val_dataset = CIFAR_10("/path/to/your/CIFAR_10", False, 224)

    elif downstream_dataset == "CIFAR100":
        num_classes = 100
        train_dataset = CIFAR_100("/path/to/your/CIFAR_100", True, 224)
        val_dataset = CIFAR_100("/path/to/your/CIFAR_100", False, 224)

    elif downstream_dataset == "SUN397":
        num_classes = 362
        train_dataset = SUN_397("/path/to/your/SUN397", True, 224)
        val_dataset = SUN_397("/path/to/your/SUN397", False, 224)

    elif downstream_dataset == "OxfordPets":
        num_classes = 37
        train_dataset = Oxford_IIIT_Pets("/path/to/your/Oxford_IIIT_Pets", True, 224)
        val_dataset = Oxford_IIIT_Pets("/path/to/your/Oxford_IIIT_Pets", False, 224)

    elif downstream_dataset == "Caltech101":
        num_classes = 101
        train_dataset = Caltech_101("/path/to/your/caltech-101", True, 224)
        val_dataset = Caltech_101("/path/to/your/caltech-101", False, 224)

    if downstream_task == "LP":
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True,
                                                   num_workers=NUM_WORKERS)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True,
                                                 num_workers=NUM_WORKERS)
        classifier = torch.nn.Sequential(
            torch.nn.Linear(768, num_classes),  # vision_width = 768
        ).to(device)
        optimizer = torch.optim.AdamW(
            [
                {"params": classifier.parameters(), "lr": 1e-3, "weight_decay": 0., "betas": (0.9, 0.999)},
            ],
            eps=1e-6,
        )
        criterion = torch.nn.BCEWithLogitsLoss()  # combined sigmoid

        for epoch in range(50):
            model.train()
            classifier.train()
            data_iter = iter(train_loader)
            for _ in tqdm(range(len(data_iter))):
                _, image, label = next(data_iter)

                optimizer.zero_grad()
                imgs = image.to(device)
                label = label.to(device)

                image_features = model.encode_image_featExt(imgs)

                probs = classifier(image_features)
                loss = criterion(probs, label)

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                model.eval()
                classifier.eval()
                data_iter = iter(val_loader)
                preds = []
                labels = []
                for _ in tqdm(range(len(data_iter))):
                    _, image, label = next(data_iter)

                    imgs = image.to(device)

                    image_features = model.encode_image_featExt(imgs)

                    pred = classifier(image_features)
                    preds.append(pred)
                    labels.append(label)

            preds = torch.cat(preds, dim=0).cpu().numpy()
            labels = torch.cat(labels, dim=0).cpu().numpy()
            auc = metrics.roc_auc_score(labels, preds)
            map = metrics.average_precision_score(labels, preds)

            print(f"Epoch {epoch + 1}: AUC = {auc}, mAP = {map}, Loss = {loss}")
            logging.info(
                f"MultiLabelCls Validation Training(epoch {epoch + 1}) | "
                f"Valid Loss: {loss:.6f} | "
                f"Valid AUC: {auc:.6f} | "
                f"Valid_MAP: {map:.6f}"
            )
        return epoch_main, auc, map, loss

    if downstream_task == "FFT":
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True,
                                                   num_workers=NUM_WORKERS)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True,
                                                 num_workers=NUM_WORKERS)
        classifier = torch.nn.Sequential(
            torch.nn.Linear(768, num_classes),  # vision_width = 768
        ).to(device)

        exclude = lambda n: "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n: not exclude(n)
        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "lr": 1e-6, "weight_decay": 0., "betas": (0.9, 0.999)},
                {"params": rest_params, "lr": 1e-6, "weight_decay": 0.001, "betas": (0.9, 0.999)},
                {"params": classifier.parameters(), "lr": 1e-3, "weight_decay": 0., "betas": (0.9, 0.999)},
            ],
            eps=1e-6,
        )
        criterion = torch.nn.BCEWithLogitsLoss()  # combined sigmoid

        for epoch in range(50):
            model.train()
            classifier.train()
            data_iter = iter(train_loader)
            for _ in tqdm(range(len(data_iter))):
                _, image, label = next(data_iter)

                optimizer.zero_grad()
                imgs = image.to(device)
                label = label.to(device)

                image_features = model.encode_image_featExt(imgs)

                probs = classifier(image_features)
                loss = criterion(probs, label)

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                model.eval()
                classifier.eval()
                data_iter = iter(val_loader)
                preds = []
                labels = []
                for _ in tqdm(range(len(data_iter))):
                    _, image, label = next(data_iter)

                    imgs = image.to(device)

                    image_features = model.encode_image_featExt(imgs)

                    pred = classifier(image_features)
                    preds.append(pred)
                    labels.append(label)

            preds = torch.cat(preds, dim=0).cpu().numpy()
            labels = torch.cat(labels, dim=0).cpu().numpy()
            auc = metrics.roc_auc_score(labels, preds)
            map = metrics.average_precision_score(labels, preds)

            print(f"Epoch {epoch + 1}: AUC = {auc}, mAP = {map}, Loss = {loss}")
            logging.info(
                f"MultiLabelCls Validation Training(epoch {epoch + 1}) | "
                f"Valid Loss: {loss:.6f} | "
                f"Valid AUC: {auc:.6f} | "
                f"Valid_MAP: {map:.6f}"
            )
        return epoch_main, auc, map, loss

    elif downstream_task == "ZSC":
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True,
                                                 num_workers=NUM_WORKERS)
        labeltext = val_loader.dataset.labeltext()
        model.eval()
        data_iter = iter(val_loader)
        acc = 0
        total = 0
        labels_total = []
        probs_total = []

        with torch.no_grad():
            for _ in tqdm(range(len(data_iter))):
                _, image, label = next(data_iter)

                texts = tokenize(labeltext)

                imgs = image.to(device)
                texts = texts.to(device)

                logits_per_image, logits_per_text = model.get_similarity(imgs, texts, onlyclip)
                probs = logits_per_image.softmax(dim=-1).cpu()

                rows, cols = label.nonzero(as_tuple=True)

                num_categories_per_sample = torch.bincount(rows, minlength=label.size(0))

                k = num_categories_per_sample.max().item()  
                topk_probs, topk_indices = torch.topk(probs, k, dim=1)
                num_samples = imgs.shape[0]
                for i in range(num_samples):
                    gt = cols[rows == i]
                    k_i = num_categories_per_sample[i]
                    pred_i = topk_indices[i, :k_i]
                    acc += torch.isin(pred_i, gt).sum() / k_i
                total += num_samples
                labels_total.append(label)
                probs_total.append(probs)

            acc_total = acc / total
            probs_total = torch.cat(probs_total, dim=0).cpu().numpy()
            labels_total = torch.cat(labels_total, dim=0).cpu().numpy()
            auc = metrics.roc_auc_score(labels_total, probs_total)
            map = metrics.average_precision_score(labels_total, probs_total)

        return epoch_main, acc_total, auc, map

    elif downstream_task == "ZSC_percategory":
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True,
                                                 num_workers=NUM_WORKERS)
        labeltext = val_loader.dataset.labeltext()
        model.eval()
        data_iter = iter(val_loader)
        acc = 0
        total = 0
        labels_total = []
        probs_total = []

        with torch.no_grad():
            for _ in tqdm(range(len(data_iter))):
                _, image, label = next(data_iter)

                texts = tokenize(labeltext)

                imgs = image.to(device)
                texts = texts.to(device)

                logits_per_image, logits_per_text = model.get_similarity(imgs, texts, onlyclip)
                probs = logits_per_image.softmax(dim=-1).cpu()

                rows, cols = label.nonzero(as_tuple=True)

                num_categories_per_sample = torch.bincount(rows, minlength=label.size(0))

                k = num_categories_per_sample.max().item() 
                topk_probs, topk_indices = torch.topk(probs, k, dim=1)
                num_samples = imgs.shape[0]
                for i in range(num_samples):
                    gt = cols[rows == i]
                    k_i = num_categories_per_sample[i]
                    pred_i = topk_indices[i, :k_i]
                    acc += torch.isin(pred_i, gt).sum() / k_i
                total += num_samples
                labels_total.append(label)
                probs_total.append(probs)

        probs_total = torch.cat(probs_total, dim=0).cpu().numpy()
        labels_total = torch.cat(labels_total, dim=0).cpu().numpy()

        acc_total = acc / total

        num_classes = labels_total.shape[1]
        per_class_auc = []
        per_class_ap = []

        for i in range(num_classes):
            try:
                auc_i = metrics.roc_auc_score(labels_total[:, i], probs_total[:, i])
            except ValueError:
                auc_i = float('nan')
            try:
                ap_i = metrics.average_precision_score(labels_total[:, i], probs_total[:, i])
            except ValueError:
                ap_i = float('nan')
            per_class_auc.append(auc_i)
            per_class_ap.append(ap_i)

        return epoch_main, acc_total, per_class_auc, per_class_ap

    if downstream_task == "LP_percategory":
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True,
                                                   num_workers=NUM_WORKERS)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True,
                                                 num_workers=NUM_WORKERS)
        classifier = torch.nn.Sequential(
            torch.nn.Linear(768, num_classes),  # vision_width = 768
        ).to(device)
        optimizer = torch.optim.AdamW(
            [
                {"params": classifier.parameters(), "lr": 1e-3, "weight_decay": 0., "betas": (0.9, 0.999)},
            ],
            eps=1e-6,
        )
        criterion = torch.nn.BCEWithLogitsLoss()  # combined sigmoid

        for epoch in range(50):
            model.train()
            classifier.train()
            data_iter = iter(train_loader)
            for _ in tqdm(range(len(data_iter))):
                _, image, label = next(data_iter)

                optimizer.zero_grad()
                imgs = image.to(device)
                label = label.to(device)

                image_features = model.encode_image_featExt(imgs)

                probs = classifier(image_features)
                loss = criterion(probs, label)

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                model.eval()
                classifier.eval()
                data_iter = iter(val_loader)
                preds = []
                labels = []
                for _ in tqdm(range(len(data_iter))):
                    _, image, label = next(data_iter)

                    imgs = image.to(device)

                    image_features = model.encode_image_featExt(imgs)

                    pred = classifier(image_features)
                    preds.append(pred)
                    labels.append(label)

            preds = torch.cat(preds, dim=0).cpu().numpy()  # shape: [num_samples, num_classes]
            labels = torch.cat(labels, dim=0).cpu().numpy()  # shape: [num_samples, num_classes]

            auc = metrics.roc_auc_score(labels, preds)
            map = metrics.average_precision_score(labels, preds)

            num_classes = labels.shape[1]
            per_class_auc = []
            per_class_ap = []

            for i in range(num_classes):
                try:
                    auc_i = metrics.roc_auc_score(labels[:, i], preds[:, i])
                except ValueError:
                    auc_i = float('nan')
                try:
                    ap_i = metrics.average_precision_score(labels[:, i], preds[:, i])
                except ValueError:
                    ap_i = float('nan')
                per_class_auc.append(auc_i)
                per_class_ap.append(ap_i)


            print(f"Epoch {epoch + 1}: AUC = {auc}, mAP = {map}, Loss = {loss}")
            logging.info(
                f"MultiLabelCls Validation Training(epoch {epoch + 1}) | "
                f"Valid Loss: {loss:.6f} | "
                f"Valid AUC: {per_class_auc} | "
                f"Valid_MAP: {per_class_ap}"
            )
        return epoch_main, auc, map, loss

    else:
        return -1
    