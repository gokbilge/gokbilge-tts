import piper_train
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
print("pytorch_lightning OK, version:", __import__("pytorch_lightning").__version__)
from piper_train.vits.lightning import VitsModel
print("VitsModel OK")
print("piper_train fully importable")
