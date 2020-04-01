from data.datasets.bases import BaseImageDataset


class MultiDataset(BaseImageDataset):
    def __init__(self, datasets):
        self._datasets = datasets

        self._train = None
        self._query = None
        self._gallery = None

    @property
    def num_train_pids(self):
        return sum([ds.num_train_pids for ds in self._datasets])

    @property
    def num_train_imgs(self):
        return sum([ds.num_train_imgs for ds in self._datasets])

    @property
    def num_train_cams(self):
        return sum([ds.num_train_cams for ds in self._datasets])

    @property
    def num_query_pids(self):
        return sum([ds.num_query_pids for ds in self._datasets])

    @property
    def num_query_imgs(self):
        return sum([ds.num_query_imgs for ds in self._datasets])

    @property
    def num_query_cams(self):
        return sum([ds.num_query_cams for ds in self._datasets])

    @property
    def num_gallery_pids(self):
        return sum([ds.num_gallery_pids for ds in self._datasets])

    @property
    def num_gallery_imgs(self):
        return sum([ds.num_gallery_imgs for ds in self._datasets])

    @property
    def num_gallery_cams(self):
        return sum([ds.num_gallery_cams for ds in self._datasets])

    @property
    def train(self):
        if self._train is None:
            self._train = [item for ds in self._datasets for item in ds.train]
        return self._train

    @property
    def query(self):
        if self._query is None:
            self._query = [item for ds in self._datasets for item in ds.query]
        return self._query

    @property
    def gallery(self):
        if self._gallery is None:
            self._gallery = [item for ds in self._datasets for item in ds.gallery]
        return self._gallery
