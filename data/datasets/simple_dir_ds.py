from pathlib import Path
from typing import List

from .bases import BaseImageDataset


class SimpleDirDs(BaseImageDataset):
    ds_name = 'rzd_reid'

    def __init__(self, root='./', datasets=('items', 'persons'),
                 verbose=True, **kwargs):
        super(SimpleDirDs, self).__init__(**kwargs)
        self.datasets_dirs: List[Path] = [Path(root) / self.ds_name / ds for ds in datasets]

        self.train_dir = ''
        self.query_dir = ''
        self.gallery_dir = ''

        train = self._process_dir()
        query = []
        gallery = []

        if verbose:
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _process_dir(self):
        camid = 0
        pid = 0

        dataset = []
        for ds_paths in self.datasets_dirs:
            assert ds_paths.exists()
            for imgs_path in ds_paths.iterdir():
                for img_path in filter(lambda x: x.suffix.lower() in ('.jpg', '.png'), imgs_path.iterdir()):
                    dataset.append((str(img_path.absolute()), pid, camid))
                pid += 1

        return self._offset_recalc(dataset, 'train')
