"""Patch piper_train/vits/lightning.py for pytorch-lightning 2.x.

Changes:
- Add self.automatic_optimization = False to VitsModel.__init__
- Rewrite training_step to use manual optimization (optimizer_idx arg removed in 2.x)
- Add on_train_epoch_end for LR scheduler stepping
"""
import re
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    src = f.read()

# 1. Add automatic_optimization = False after save_hyperparameters()
src = src.replace(
    "        self.save_hyperparameters()\n",
    "        self.save_hyperparameters()\n"
    "        self.automatic_optimization = False  # lightning 2.x: multiple optimizers require manual opt\n",
    1,
)

# 2. Replace training_step (optimizer_idx style) with manual optimization version
old_step = '''\
    def training_step(self, batch: Batch, batch_idx: int, optimizer_idx: int):
        if optimizer_idx == 0:
            return self.training_step_g(batch)

        if optimizer_idx == 1:
            return self.training_step_d(batch)'''

new_step = '''\
    def training_step(self, batch: Batch, batch_idx: int):
        opt_g, opt_d = self.optimizers()

        # Generator step
        opt_g.zero_grad()
        loss_g = self.training_step_g(batch)
        self.manual_backward(loss_g)
        opt_g.step()

        # Discriminator step
        opt_d.zero_grad()
        loss_d = self.training_step_d(batch)
        self.manual_backward(loss_d)
        opt_d.step()

    def on_train_epoch_end(self):
        sch_g, sch_d = self.lr_schedulers()
        sch_g.step()
        sch_d.step()'''

if old_step not in src:
    print("ERROR: could not find training_step to patch", file=sys.stderr)
    sys.exit(1)

src = src.replace(old_step, new_step, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(src)

print("Patched:", path)
