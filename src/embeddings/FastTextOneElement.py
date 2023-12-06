"""Fasttext model learning on elements of rows in dataset."""
from gensim.models.fasttext import FastText
from torch.utils.data import DataLoader
from src.data_process.data_classes import AllCsvDataSet
from gensim.models.callbacks import CallbackAny2Vec
from datetime import datetime
import os


class LossLogger(CallbackAny2Vec):
    """Loss logger for 2vec models."""
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        else:
            print('Loss after epoch {}: {}'.format(
                self.epoch,
                loss - self.loss_previous_step
                ))
        self.epoch += 1
        self.loss_previous_step = loss


class FastTextOneElement:
    """Fasttext model class. Learning on elements."""
    def __init__(self, data_dir: str) -> None:
        """data_dit -- directory with datasets."""
        dataset = AllCsvDataSet(data_dir=data_dir, return_row=False)
        print('Data len:', len(dataset))
        self.dataloader = DataLoader(dataset,
                                     batch_size=1,
                                     shuffle=True)

        self.model = FastText(vector_size=256,
                              window=3,
                              sg=1,
                              hs=1,
                              seed=42,
                              negative=12,
                              min_n=2,
                              max_n=4,
                              workers=6,
                              min_count=0,
                              epochs=200)

    def train(self, epochs: int = 200, save_dir_pth: str = 'models/'):
        """Train and save model."""
        print('Start build vocabulary')
        self.model.build_vocab(self.dataloader)
        print('Corpus count:', self.model.corpus_count)
        print('Corpus total words:', self.model.corpus_total_words)
        print('Total words vectors len:', len(self.model.wv))
        print('Dataloader len:', len(self.dataloader))
        self.model.train(
            corpus_iterable=self.dataloader,
            total_examples=len(self.dataloader),
            epochs=epochs,
            compute_loss=True,
            callbacks=[LossLogger()]
        )
        if save_dir_pth:
            self.save(save_dir_pth)

    def save(self, save_dir_pth: str = 'models/'):
        """Save model."""
        save_pth = os.path.join(save_dir_pth, 'fasttext_one_element' +
                                datetime.now().strftime("_%y%m%d-%H%M%S") +
                                '.model')
        self.model.save(save_pth)
        print(f'Model saved to {save_pth}')
