import numpy as np
import torch
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) ])

print('Training on',DEVICE)

# init model and load final state
from ml_lightcurve.cvae.model_cvae import Kamile_CVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.cuda.empty_cache()



def load_model(model_dir, model_metada):
    """ loads the model (eval mode) with all dictionaries """
    model = Kamile_CVAE.init_from_dict(model_metada)
    state = torch.load(model_dir+"model.pt", map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    model.to(device)
    return (model, state)


