import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torchvision.transforms import functional as TF
from einops import rearrange, repeat
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import pytorch_ssim
import torchvision.utils as utils
import torchvision.utils as vutils

DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")

#1) Vision transformer


class Residual(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn=fn
    def forward(self,x,**kwargs):
        return self.fn(x,**kwargs)+x
class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self,x,**kwargs):
        x=self.norm(x)
        return self.fn(x,**kwargs)

class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,dim)
        )
    def forward(self,x):
        return self.out(x)


class Attention(nn.Module):
    def __init__(self,dim,heads):
        super().__init__()
        self.heads = heads
        self.scale = dim**-0.5
        self.to_qkv = nn.Linear(dim,dim*3,bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim,dim),
        )
    def forward(self,x,mask=None):
        b,n,_,h = *x.shape,self.heads
        qkv=self.to_qkv(x).chunk(3,dim=-1)
        q,k,v = map(lambda a:rearrange(a,'b n (h d) -> b h n d',h=h),qkv)
        dots = th.einsum('bhid,bhjd->bhij',q,k)#does a dot product for each q and k for every head; similarities between queries and keys
        mask_val = -th.finfo(dots.dtype).max #gives the largest -ve number of dots.dype
        if mask is not None:
            mask = F.pad(mask.flatten(1),(1,0),value=True)  #adds on padding element with value true to the left 
            assert mask.shape[-1]==dots.shape[-1],"mask has wrong dims"
            mask = mask[:,None,:]*mask[:,:,None]
            dots.masked_fill_(~mask,mask_val)
            del mask
        attn = dots.softmax(dim=-1)
        out=th.einsum('bhij,bhjd->bhid',attn,v)
        out = rearrange(out,'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
            

class Transformer(nn.Module):
    def __init__(self,dim,depth,heads,mlp_dim):
        super(Transformer,self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim,Attention(dim,heads))),
                Residual(PreNorm(dim,FeedForward(dim,mlp_dim)))
            ]))
    def forward(self,x,mask=None):
        for attn, ff in self.layers:
            x = attn(x,mask = mask)
            x = ff(x)
        return x
    

class ViT(nn.Module):
    def __init__(self,*,image_size, patch_size, num_classes,dim,depth,heads,mlp_dim,channels=3):
        super(ViT,self).__init__()
        assert image_size%patch_size==0, "image not divisible by patch size"
        num_patches = (image_size//patch_size)**2
        patch_dim = channels*patch_size**2
        self.patch_size=patch_size
        self.pos_embedding=nn.Parameter(th.randn(1,num_patches+1,dim)) #+1 for cls token
        self.flattened_patch=nn.Linear(patch_dim,dim)
        self.cls_token = nn.Parameter(th.randn(1,1,dim))
        self.transformer = Transformer(dim,depth,heads,mlp_dim)
        self.to_cls_token=nn.Identity()
        

    def forward(self,img,mask=None):
        p=self.patch_size
        x = rearrange(img,'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.flattened_patch(x)
        b,n,_=x.shape
        cls_token = repeat(self.cls_token,'() n d -> b n d',b=b)
        x = th.cat((cls_token,x),dim=1)
        x += self.pos_embedding[:,:(n+1)]
        x = self.transformer(x,mask)
        x = self.to_cls_token(x[:,1:,:]) #im not removing the class token but extracting the transformational representation of the image
        return x
    
 #Decoder
class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.decoder2 = nn.Sequential(
             nn.ConvTranspose2d(in_channels= in_channels, out_channels=16,kernel_size= 3, stride=2,padding=1),  # In b, 8, 8, 8 >> out b, 16, 15, 15
             nn.BatchNorm2d(16, affine = True),
             nn.ReLU(True),            
             nn.ConvTranspose2d(16, 32, 9, stride=3, padding = 1),  #out> b,32, 49, 49
             nn.BatchNorm2d(32, affine = True),
             nn.ReLU(True),             
             nn.ConvTranspose2d(32, 32, 7, stride=5, padding=1),  #out> b, 32, 245, 245
             nn.BatchNorm2d(32, affine = True),
             nn.ReLU(True), 
             nn.ConvTranspose2d(32, 16, 9, stride=2),  #out> b, 16, 497, 497
             nn.BatchNorm2d(16, affine = True),
             nn.ReLU(True), 
             nn.ConvTranspose2d(16, 8, 6, stride=1),  #out> b, 8, 502, 502
             nn.BatchNorm2d(8, affine = True),
             nn.ReLU(True),
             nn.ConvTranspose2d(8, 3, 11, stride=1),  #out> b, 3, 512, 512
             nn.Tanh()
             )
       

    def forward(self, x):
         recon = self.decoder2(x)
         return recon

   
class DigitCaps(nn.Module):
    def __init__(self, out_num_caps=1, in_num_caps=8*8*64, in_dim_caps=8, out_dim_caps=512, decode_idx=-1):
        super(DigitCaps, self).__init__()
        self.in_dim_caps = in_dim_caps
        self.in_num_caps = in_num_caps
        self.out_dim_caps = out_dim_caps
        self.out_num_caps = out_num_caps
        self.decode_idx = decode_idx
        self.W = nn.Parameter(0.01 * th.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))
#        self.upsample = upsampling()

    def forward(self, x):
        # x size: batch x 1152 x 8
        x_hat = th.squeeze(th.matmul(self.W, x[:, None, :, :, None]), dim=-1)
        x_hat_detached = x_hat.detach()
        # x_hat size: batch x ndigits x 1152 x 16
        b = th.zeros(x.size(0), self.out_num_caps, self.in_num_caps)
        # b size: batch x ndigits x 1152
        
        b = b.to(DEVICE)

        # routing algo taken from https://github.com/XifengGuo/CapsNet-Pytorch/blob/master/capsulelayers.py
        num_iters = 3
        for i in range(num_iters):
            c = F.softmax(b, dim=1)
            # c size: batch x ndigits x 1152
            if i == num_iters -1:
                # output size: batch x ndigits x 1 x 16
                outputs = self.squash(th.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
            else:
                outputs = self.squash(th.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                b = b + th.sum(outputs * x_hat_detached, dim=-1)


        outputs = th.squeeze(outputs, dim=-2) # squeezing to remove ones at the dimension -1
        # Below code chooses the maximum lenth of the vector
        if self.decode_idx == -1:  # choose the longest vector as the one to decode
            classes = th.sqrt((outputs ** 2).sum(2))
            classes = F.softmax(classes, dim=1)
            _, max_length_indices = classes.max(dim=1)
        else:  # always choose the same digitcaps
            max_length_indices = th.ones(outputs.size(0)).long() * self.decode_idx
            
            max_length_indices = max_length_indices.to(DEVICE)

        masked = th.sparse.torch.eye(self.out_num_caps)
        
        masked = masked.to(DEVICE)
        masked = masked.index_select(dim=0, index=max_length_indices)
#        t = (outputs * masked[:, :, None]).view(x.size(0), -1)
        t = (outputs * masked[:, :, None]).sum(dim=1).unsqueeze(1)
#        t = self.upsample(t)

        return t, outputs

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * th.sqrt(squared_norm))
        return output_tensor
    
#Mixture density Network
class MDN(nn.Module):

    def __init__(self, input_dim, out_dim, layer_size, coefs, test=False, sd=0.5):
        super(MDN, self).__init__()
        self.in_features = input_dim

        self.pi = nn.Linear(layer_size, coefs, bias=False)
        self.mu = nn.Linear(layer_size, out_dim * coefs, bias=False)  # mean
        self.sigma_sq = nn.Linear(layer_size, out_dim * coefs, bias=False)  # isotropic independent variance
        self.out_dim = out_dim
        self.coefs = coefs
        self.test = test
        self.sd = sd

    def forward(self, x):
        ep = np.finfo(float).eps
        x = th.clamp(x, ep)

        pi = F.softmax(self.pi(x), dim=-1)
        sigma_sq = F.softplus(self.sigma_sq(x)).view(x.size(0),x.size(1),self.in_features, -1)  # logvar
        mu = self.mu(x).view(x.size(0),x.size(1),self.in_features, -1)  # mean
        return pi, mu, sigma_sq

#Loss Fn

def log_gaussian(x, mean, logvar):
    
    x = x.unsqueeze(-1).expand_as(logvar)
    a = (x - mean) ** 2  # works on multiple samples thanks to tensor broadcasting
    log_p = (logvar + a / (th.exp(logvar))).sum(2)
    log_p = -0.5 * (np.log(2 * np.pi) + log_p)
    
    return log_p 


def log_gmm(x, means, logvars, weights, total=True):
    res = -log_gaussian(x ,means, logvars) # negative of log likelihood
    
    res = weights * res

    if total:
        return th.sum(res,2)
    else:
        return res


def mdn_loss_function(x, means, logvars, weights, test=False):
    if test:
        res = log_gmm(x, means, logvars, weights)
    else:
        res = th.mean(th.sum(log_gmm(x, means, logvars, weights),1))
    return res

def add_noise(latent, noise_type="gaussian", sd=0.2):
    
    assert sd >= 0.0
    if noise_type == "gaussian":
        mean = 0.

        n = th.distributions.Normal(th.tensor([mean]), th.tensor([sd]))
        noise = n.sample(latent.size()).squeeze(-1).to(DEVICE)
        latent = latent + noise
        return latent

    if noise_type == "speckle":
        noise = th.randn(latent.size()).to(DEVICE)
        latent = latent + latent * noise
        return latent

class autoencoder(nn.Module):
    def __init__(self,
                 image_size = 512,
                 patch_size = 64,
                 num_classes = 1,
                 dim = 512,
                 depth=6,
                 heads = 8,
                 mlp_dim=1024,
                 train=True
                 ):
        super().__init__()
        self.vt = ViT(
            image_size = image_size,
            patch_size = patch_size,
            num_classes = num_classes,
            dim = dim,
            depth = depth,
            heads = heads,
            mlp_dim = mlp_dim )
        self.decoder=Decoder(8)
        self.Digicap = DigitCaps(in_num_caps=((image_size//patch_size)**2)*8*8, in_dim_caps=8)
        self.train = train
        if train:
            self.initialize_weights(self.vt,self.decoder)
        self.mask = th.ones(1, image_size//patch_size, image_size//patch_size).bool().to(DEVICE)


    def initialize_weights(*models):
        for model in models:
            for module in model.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()
    
    def forward(self,x):
        b = x.shape[0]
        encoded = self.vt(x,self.mask)
        if self.train:
            encoded = add_noise(encoded)
        encoded1,vectors = self.Digicap(encoded.view(b,encoded.size(1)*8*8,-1))
        recons = self.decoder(encoded1.view(b,-1,8,8))
        return encoded, recons


class BTechDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('.bmp')]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image

transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
])

dataset = BTechDataset(root_dir="/N/slate/aalefern/image/BTech_Dataset_transformed/01/train/ok", transform=transform)
train_loader = th.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

model = autoencoder(patch_size=64,train=True).to(DEVICE)
mdn = MDN(input_dim=512, out_dim=512, layer_size=512, coefs=10).to(DEVICE)
Optimiser = Adam(list(model.parameters())+list(mdn.parameters()), lr=0.0001, weight_decay=0.0001)
from torchmetrics.image import StructuralSimilarityIndexMeasure

ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0) 
epoch = 400


loss_list = []
print('\nNetwork training started.....')
for i in range(epoch):
    t_loss = []
    
    for j in train_loader:
        if j.size(1)==1:
            j = th.stack([j,j,j]).squeeze(2).permute(1,0,2,3)
        model.zero_grad()
        
        # vector,pi, mu, sigma, reconstructions = model(j.cuda())
        vector, reconstructions = model(j.to(DEVICE))
        pi, mu, sigma = mdn(vector)
        
        #Loss calculations
        loss1 = F.mse_loss(reconstructions,j.to(DEVICE), reduction='mean') #Rec Loss
        loss2 = -ssim_loss(j.to(DEVICE), reconstructions) #SSIM loss for structural similarity
        loss3 = mdn_loss_function(vector,mu,sigma,pi) #MDN loss for gaussian approximation
        
        
        loss = 5*loss1 + 0.5*loss2 + loss3       #Total loss
        
        t_loss.append(loss.detach().cpu().item())   #storing all batch losses to calculate mean epoch loss
        

        #Optimiser step
        loss.backward()
        Optimiser.step()
    
    img_grid = vutils.make_grid(reconstructions).permute(1, 2, 0).cpu().numpy()
    loss_list.append(sum(t_loss)/len(t_loss))
    
    plt.imsave(f'reconstructed_epoch_{i}.png', img_grid)
    plt.close()
    # Create the plot
    plt.plot(t_loss, marker='o', linestyle='-')

    # Labels and title
    plt.xlabel('steps')
    plt.ylabel('T_loss')
    plt.title('Plot of t_loss Values')

    # Save the figure instead of displaying it
    plt.savefig(f'plot_t_loss_{i}.png', dpi=300, bbox_inches='tight')

    # Close the figure to free up memory
    plt.close()


# Create the plot
plt.plot(t_loss, marker='o', linestyle='-')

# Labels and title
plt.xlabel('epochs')
plt.ylabel('avg_T_loss')
plt.title('Plot of avg_t_loss Values')

# Save the figure instead of displaying it
plt.savefig(f'plot_avg_t_loss_{i}.png', dpi=300, bbox_inches='tight')

# Close the figure to free up memory
plt.close()


    