import torch
import lightning as L

class BaseModel(L.LightningModule):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)
        # print(x_hat.shape,y.shape)
        loss = F.mse_loss(x_hat, y)
        self.log("train_loss", loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, y)
        self.log("val_loss", loss,prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer