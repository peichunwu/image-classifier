from torch import Tensor
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

input_size = 224
data_dir = './data'

class dataset_voc(torchvision.datasets.VOCDetection):

    def __init__(self, root, trvaltest, transform = None):
        """
            Parameters:
                root (string) – Root directory of the VOC Dataset.
                year (string, optional) – The dataset year, supports years "2007" to "2012".
                image_set (string, optional) – Select the image_set to use, "train", "trainval" or "val". If year=="2007", can also be "test".
                download (bool, optional) – If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. (default: alphabetic indexing of VOC’s 20 classes).
                transform (callable, optional) – A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop
                target_transform (callable, required) – A function/transform that takes in the target and transforms it.
                transforms (callable, optional) – A function/transform that takes input sample and its target as entry and returns a transformed version.
        """
        super(dataset_voc, self).__init__(root, year = '2012', image_set = trvaltest,
                                          download = False, transform = transform)

    def list_image_sets(self):
        """
            Summary:
                List all the image sets from Pascal VOC. Don't bother computing
                this on the fly, just remember it. It's faster.
        """
        return [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']

    def __getitem__(self, idx):
        image, target = super(dataset_voc, self).__getitem__(idx)
        cls = [i["name"] for i in target['annotation']['object']]
        label = Tensor([1 if s in cls else 0 for s in self.list_image_sets()])

#         sample = {'image': image, 'label': Tensor.new_tensor(label),
#                   'filename': item[1]['annotation']['filename']}

        return image, label
