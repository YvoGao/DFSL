import os
import pickle
import math
import random
from collections import defaultdict
from .utils import Datum, DatasetBase, read_json, mkdir_if_missing, write_json

class CUB200(DatasetBase):
    # CUB200数据集根目录下的核心文件夹名
    dataset_dir = "CUB_200_2011"

    def __init__(self, cfg):
        # 处理根路径（兼容配置文件传参）
        root = os.path.abspath(os.path.expanduser(cfg.dataset_root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_CUB200.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        # 保留原有属性
        self.classes = CLASSES
        self.gpt_prompt_path = None
        # 读取/生成数据集划分（train/val/test）
        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            # 读取原始train数据作为trainval，再拆分train/val
            trainval = self.read_data(split_type="train")
            test = self.read_data(split_type="test")
            train, val = self.split_trainval(trainval)
            self.save_split(train, val, test, self.split_path, self.image_dir)

        # 处理Few-shot逻辑（和oxford_pets对齐）
        num_shots = cfg.num_shots
        if num_shots >= 1:
            seed = 1
            preprocessed = os.path.join(
                self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl"
            )
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file)

        # 处理类别子采样（和oxford_pets对齐）
        subsample = cfg.subsample_classes
        train, val, test = self.subsample_classes(train, val, test, subsample=subsample)

        # 初始化父类（DatasetBase）
        super().__init__(train_x=train, val=val, test=test)

        

    def read_data(self, split_type):
        """
        读取CUB200原始标注并转换为Datum格式
        split_type: "train" 读取原训练集（作为trainval） | "test" 读取原测试集
        """
        # 读取核心标注文件
        image_file = os.path.join(self.dataset_dir, "images.txt")
        split_file = os.path.join(self.dataset_dir, "train_test_split.txt")
        class_file = os.path.join(self.dataset_dir, "image_class_labels.txt")

        # 解析标注文件为字典
        id2image = self.list2dict(self.text_read(image_file))
        id2train = self.list2dict(self.text_read(split_file))  # 1=train, 0=test
        id2class = self.list2dict(self.text_read(class_file))

        items = []
        for img_id in id2image.keys():
            # 判断是否属于目标划分
            is_train_split = id2train[img_id] == "1"
            if (split_type == "train" and is_train_split) or (split_type == "test" and not is_train_split):
                # 构建图片路径和标签
                img_path = os.path.join(self.image_dir, id2image[img_id])
                label = int(id2class[img_id]) - 1  # 转为0-base
                classname = self.classes[label]
                items.append(Datum(impath=img_path, label=label, classname=classname))
        return items

    def text_read(self, file):
        """保留原有文件读取逻辑"""
        with open(file, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip("\n")
        return lines

    def list2dict(self, list_data):
        """优化分割逻辑：兼容带空格的图片路径"""
        dict_data = {}
        for l in list_data:
            # 仅分割第一个空格（处理路径中的空格）
            parts = l.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"无效的标注行格式: {l}")
            img_id, val = parts
            if img_id not in dict_data:
                dict_data[img_id] = val
            else:
                raise EOFError("同一个ID只能出现一次")
        return dict_data

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        """拆分trainval为train/val（和oxford_pets逻辑完全对齐）"""
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            tracker[item.label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0, f"标签{label}的样本数不足以划分验证集"
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                if n < n_val:
                    val.append(trainval[idx])
                else:
                    train.append(trainval[idx])
        return train, val

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        """保存划分结果到JSON（和oxford_pets逻辑完全对齐）"""
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath.replace(path_prefix, "")
                impath = impath.lstrip("/")
                out.append((impath, item.label, item.classname))
            return out

        split = {
            "train": _extract(train),
            "val": _extract(val),
            "test": _extract(test)
        }
        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        """从JSON读取划分结果（和oxford_pets逻辑完全对齐）"""
        def _convert(items):
            out = []
            for impath, label, classname in items:
                full_path = os.path.join(path_prefix, impath)
                out.append(Datum(impath=full_path, label=int(label), classname=classname))
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        return _convert(split["train"]), _convert(split["val"]), _convert(split["test"])

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """类别子采样（和oxford_pets逻辑完全对齐）"""
        assert subsample in ["all", "base", "new"]
        if subsample == "all":
            return args

        # 提取所有类别并排序
        dataset = args[0]
        labels = sorted({item.label for item in dataset})
        n = len(labels)
        m = math.ceil(n / 2)

        # 选择base/new类别
        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        selected = labels[:m] if subsample == "base" else labels[m:]
        relabeler = {old: new for new, old in enumerate(selected)}

        # 重新处理数据集
        output = []
        for ds in args:
            new_ds = []
            for item in ds:
                if item.label in selected:
                    new_ds.append(Datum(
                        impath=item.impath,
                        label=relabeler[item.label],
                        classname=item.classname
                    ))
            output.append(new_ds)
        return output

    @staticmethod
    def generate_fewshot_dataset(dataset, num_shots):
        """生成Few-shot数据集（和oxford_pets逻辑完全对齐）"""
        tracker = defaultdict(list)
        for idx, item in enumerate(dataset):
            tracker[item.label].append(idx)

        fewshot_ds = []
        for label, idxs in tracker.items():
            random.shuffle(idxs)
            fewshot_ds.extend([dataset[idx] for idx in idxs[:num_shots]])
        return fewshot_ds

    # ---------------- 保留原有公共方法（保证调用兼容） ----------------
    def get_class_name(self):
        return self.classes

    def get_train_data(self):
        train_data = [item.impath for item in self.train_x]
        train_targets = [item.label for item in self.train_x]
        return train_data, train_targets

    def get_test_data(self):
        test_data = [item.impath for item in self.test]
        test_targets = [item.label for item in self.test]
        return test_data, test_targets


# 保留原有类别列表（完全不变）
CLASSES = ['Black footed Albatross', 'Laysan Albatross', 'Sooty Albatross', 'Groove billed Ani', 'Crested Auklet', 
           'Least Auklet', 'Parakeet Auklet', 'Rhinoceros Auklet', 'Brewer Blackbird', 'Red winged Blackbird', 
           'Rusty Blackbird', 'Yellow headed Blackbird', 'Bobolink', 'Indigo Bunting', 'Lazuli Bunting', 'Painted Bunting', 
           'Cardinal', 'Spotted Catbird', 'Gray Catbird', 'Yellow breasted Chat', 'Eastern Towhee', 'Chuck will Widow', 'Brandt Cormorant', 
           'Red faced Cormorant', 'Pelagic Cormorant', 'Bronzed Cowbird', 'Shiny Cowbird', 'Brown Creeper', 'American Crow', 'Fish Crow', 
           'Black billed Cuckoo', 'Mangrove Cuckoo', 'Yellow billed Cuckoo', 'Gray crowned Rosy Finch', 'Purple Finch', 'Northern Flicker', 
           'Acadian Flycatcher', 'Great Crested Flycatcher', 'Least Flycatcher', 'Olive sided Flycatcher', 'Scissor tailed Flycatcher', 
           'Vermilion Flycatcher', 'Yellow bellied Flycatcher', 'Frigatebird', 'Northern Fulmar', 'Gadwall', 'American Goldfinch', 
           'European Goldfinch', 'Boat tailed Grackle', 'Eared Grebe', 'Horned Grebe', 'Pied billed Grebe', 'Western Grebe', 'Blue Grosbeak', 
           'Evening Grosbeak', 'Pine Grosbeak', 'Rose breasted Grosbeak', 'Pigeon Guillemot', 'California Gull', 'Glaucous winged Gull', 
           'Heermann Gull', 'Herring Gull', 'Ivory Gull', 'Ring billed Gull', 'Slaty backed Gull', 'Western Gull', 'Anna Hummingbird', 
           'Ruby throated Hummingbird', 'Rufous Hummingbird', 'Green Violetear', 'Long tailed Jaeger', 'Pomarine Jaeger', 'Blue Jay', 
           'Florida Jay', 'Green Jay', 'Dark eyed Junco', 'Tropical Kingbird', 'Gray Kingbird', 'Belted Kingfisher', 'Green Kingfisher', 
           'Pied Kingfisher', 'Ringed Kingfisher', 'White breasted Kingfisher', 'Red legged Kittiwake', 'Horned Lark', 'Pacific Loon', 
           'Mallard', 'Western Meadowlark', 'Hooded Merganser', 'Red breasted Merganser', 'Mockingbird', 'Nighthawk', 'Clark Nutcracker', 
           'White breasted Nuthatch', 'Baltimore Oriole', 'Hooded Oriole', 'Orchard Oriole', 'Scott Oriole', 'Ovenbird', 'Brown Pelican', 
           'White Pelican', 'Western Wood Pewee', 'Sayornis', 'American Pipit', 'Whip poor Will', 'Horned Puffin', 'Common Raven', 
           'White necked Raven', 'American Redstart', 'Geococcyx', 'Loggerhead Shrike', 'Great Grey Shrike', 'Baird Sparrow', 
           'Black throated Sparrow', 'Brewer Sparrow', 'Chipping Sparrow', 'Clay colored Sparrow', 'House Sparrow', 'Field Sparrow', 
           'Fox Sparrow', 'Grasshopper Sparrow', 'Harris Sparrow', 'Henslow Sparrow', 'Le Conte Sparrow', 'Lincoln Sparrow', 
           'Nelson Sharp tailed Sparrow', 'Savannah Sparrow', 'Seaside Sparrow', 'Song Sparrow', 'Tree Sparrow', 'Vesper Sparrow', 
           'White crowned Sparrow', 'White throated Sparrow', 'Cape Glossy Starling', 'Bank Swallow', 'Barn Swallow', 'Cliff Swallow', 
           'Tree Swallow', 'Scarlet Tanager', 'Summer Tanager', 'Artic Tern', 'Black Tern', 'Caspian Tern', 'Common Tern', 'Elegant Tern', 
           'Forsters Tern', 'Least Tern', 'Green tailed Towhee', 'Brown Thrasher', 'Sage Thrasher', 'Black capped Vireo', 'Blue headed Vireo', 
           'Philadelphia Vireo', 'Red eyed Vireo', 'Warbling Vireo', 'White eyed Vireo', 'Yellow throated Vireo', 'Bay breasted Warbler', 
           'Black and white Warbler', 'Black throated Blue Warbler', 'Blue winged Warbler', 'Canada Warbler', 'Cape May Warbler', 'Cerulean Warbler', 
           'Chestnut sided Warbler', 'Golden winged Warbler', 'Hooded Warbler', 'Kentucky Warbler', 'Magnolia Warbler', 'Mourning Warbler', 
           'Myrtle Warbler', 'Nashville Warbler', 'Orange crowned Warbler', 'Palm Warbler', 'Pine Warbler', 'Prairie Warbler', 'Prothonotary Warbler', 
           'Swainson Warbler', 'Tennessee Warbler', 'Wilson Warbler', 'Worm eating Warbler', 'Yellow Warbler', 'Northern Waterthrush', 
           'Louisiana Waterthrush', 'Bohemian Waxwing', 'Cedar Waxwing', 'American Three toed Woodpecker', 'Pileated Woodpecker', 
           'Red bellied Woodpecker', 'Red cockaded Woodpecker', 'Red headed Woodpecker', 'Downy Woodpecker', 'Bewick Wren', 'Cactus Wren', 
           'Carolina Wren', 'House Wren', 'Marsh Wren', 'Rock Wren', 'Winter Wren', 'Common Yellowthroat']