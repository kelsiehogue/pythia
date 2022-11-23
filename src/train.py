import torch
import torch.nn.functional as F
import torch.utils.data.dataloader as D
from net import PythianEngine
from dataset import Dataset
from utils import *
from sklearn.model_selection import train_test_split
import atexit
import time

def train_test_data(dataset, test_split):
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=test_split)
    return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, test_idx)

class Trainer:
    def __init__(self, nheads=8,
                       nlayers=8,
                       expansion=4,
                       kernel_size=(3,3),
                       epochs=1000,
                       max_ctx_length=24,
                       batch_size=1,
                       lr=3e-4,
                       halting_iteration=64,
                       latent_size=1024,
                       d_model=32,
            ):
        self.nlayers = nlayers
        self.expansion = expansion
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.max_ctx_length = max_ctx_length
        self.batch_size = batch_size
        self.halting_iteration = halting_iteration
        self.lr = lr
        self.latent_size = latent_size
        self.nheads = nheads

        print("Loading Nets...")
        self.engine = PythianEngine(nheads, expansion, nlayers, 3, 3, kernel_size=self.kernel_size, d_model=d_model).cuda()
        print(get_nparams(self.engine), "params in generator net.")

        self.critic_engine = PythianEngine(nheads, expansion, nlayers, 6, 3, kernel_size=self.kernel_size, d_model=d_model).cuda()
        print(get_nparams(self.critic_engine), "params in discriminator net.")
        print("Nets Loaded...")

        self.dataset = Dataset(max_ctx_length=max_ctx_length)
        self.train_dataset, self.test_dataset = train_test_data(self.dataset, .2)
        self.train_dataloader = D.DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size)
        self.test_dataloader = D.DataLoader(self.test_dataset, shuffle=True, batch_size=batch_size)

        self.optim = torch.optim.Adam(self.engine.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic_engine.parameters(), lr=self.lr)
        _, self.display, self.surface = init_camera_and_window() 

    def train(self):
        """Train the model."""
    
        print("Beginning Training...")
        running_engine_loss = 0
        running_critic_loss = 0

        self.engine.train()
        for epoch in range(self.epochs):
            print("Starting Epoch:", epoch)
            self.engine.train()
            for i, (x, y) in enumerate(self.train_dataloader):
                if i != self.halting_iteration:
                    engine_loss, critic_loss = self.training_step(x, y)
                    running_engine_loss += engine_loss
                    running_critic_loss += critic_loss
                    print("Engine Loss:", "{:3f}".format(engine_loss), "~",
                            "Avg Engine Loss:", "{:3f}".format(running_engine_loss/(i+1)), "~",
                            "Critic Engine Loss:", "{:3f}".format(critic_loss), "~",
                            "Avg Critic Loss:", "{:3f}".format(running_critic_loss/(i+1)), "~",
                            "Iterations:", i+1)
                else:
                    break
            print("End Training Epoch:", epoch)
            
            running_engine_loss = 0
            running_critic_loss = 0

            if (epoch + 1) % 2 == 0:
                print("Starting Validation Epoch.")
                self.engine.eval()
                for i, (x, y) in enumerate(self.test_dataloader):
                    if i != self.halting_iteration:
                        engine_loss = self.validation_step(x, y, i)
                        running_engine_loss += engine_loss
                        print("Engine Loss:", "{:3f}".format(engine_loss), "~",
                            "Avg Engine Validation Loss:", "{:3f}".format(running_engine_loss/(i+1)), "~",
                            "Iterations:", i+1)
                    else:
                        self.dream_step(x, y, i)
                        break
            running_engine_loss = 0

            self.save()

    def training_step(self, x, y):
        """
        One optimization step
        :param x: Input data
        :param y: Target data
        :param step: Current training step
        :return: loss
        """

        x, y = x.cuda(0), y.cuda(0)

        self.optim.zero_grad()


        latent = torch.randn(x.shape[0], self.latent_size, device=x.device)
        critic_latent = torch.randn(x.shape[0], self.latent_size, device=x.device)

        y_false = self.engine(x, latent)
        y_false = torch.sigmoid(y_false)
        
        seq_false = torch.cat([x, y_false], dim=1)
        
        critique_fake = self.critic_engine(seq_false, critic_latent)
        critique_fake = torch.sigmoid(critique_fake)

        gloss = F.mse_loss(critique_fake, torch.ones_like(critique_fake))

        gloss.backward()
        self.optim.step()

        self.critic_optim.zero_grad()

        seq = torch.cat([x, y], dim=1)

        critique_real = torch.sigmoid(self.critic_engine(seq, critic_latent))
        critique_fake = torch.sigmoid(self.critic_engine(seq_false.detach(), critic_latent))
        closs = F.mse_loss(critique_real, torch.ones_like(critique_real)) + F.mse_loss(critique_fake, torch.zeros_like(critique_fake))
        closs.backward()
        self.critic_optim.step()

        gen_loss = gloss.item()
        critic_loss = closs.item()
        return gen_loss, critic_loss
    def validation_step(self, x, y, step):
        """
        One validation step
        :param x: Input data
        :param y: Target data
        :param step: Current training step
        :return: loss
        """
        x, y = x.cuda(), y.cuda()
        y_false = self.engine(x, torch.randn(x.shape[0], self.latent_size, device=y.device))
        y_false = torch.sigmoid(y_false).detach()

        loss = F.l1_loss(y_false, y)

        y_seq = torch.cat([y, y_false], 2)
        for i in y_seq[0].unsqueeze(0).split(1, -1):
            show_tensor(i.cpu().squeeze(-1), self.display, self.surface)
            time.sleep(1./24.)

        return loss.item()

    def dream_step(self, x, y, step):
        """
        One dream step
        :param x: Input data
        :param y: Target data
        :param step: Current training step
        :return: loss
        """
        x, y = x.cuda(), y.cuda()

        memory = [x.split(1, -1)[0]]
        latent = torch.randn(x.shape[0], self.latent_size, device=x.device)
        for i in range(1, x.shape[-1] + 1):
            mem_cat = torch.cat(memory, -1)
            y_fake = self.engine(mem_cat, latent)
            y_fake = torch.sigmoid(y_fake).detach()
            memory.append(y_fake.split(1, -1)[-1])
            show_tensor(memory[-1][0].unsqueeze(0).cpu().squeeze(-1), self.display, self.surface)
            time.sleep(1./24.)
    def save(self, path='../saves/checkpoint.pt'):
        """Save the model to disk."""
        torch.save({
            'optim':self.optim.state_dict(),
            'engine':self.engine.state_dict(),
            }, path)

    def load(self, path='../saves/checkpoint.pt'):
        """Load the model from disk."""

        checkpoint = torch.load(path, map_location='cpu')
        self.engine.load_state_dict(checkpoint['engine'])
        del checkpoint['engine']
        self.optim.load_state_dict(checkpoint['optim'])
        del checkpoint['optim']
        
if __name__ == '__main__':
    trainer = Trainer()
    atexit.register(lambda:trainer.save())
    trainer.train()