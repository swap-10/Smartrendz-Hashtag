import json
import torch
from torch import nn
import torchvision.transforms as transforms
import gradio as gr

with open("tags.json", "r") as fp:
  tag_dict = json.load(fp)

ind_to_tag = {v: k for k, v in tag_dict.items()}

class ViTClassifier(nn.Module):
  def __init__(self, model):
    super(ViTClassifier, self).__init__()
    self.vit = model
    
  def forward(self, img):
    vit_image_classification_output = self.vit(img)
    logits = vit_image_classification_output.logits
    return logits

model_new = torch.load("hashtag_generator_model.pt", map_location=torch.device("cpu"))
model_new.eval()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_new.to(device)

def predict_hashtags(img, topk=10):
  transforms_list = transforms.Compose([
      transforms.Resize(size=(224, 224)),
      transforms.ToTensor()
  ])
  img = transforms_list(img)
  if img.shape[0] == 4:
    img = img[:3, :, :]
  img = img.unsqueeze(dim=0).to(device)
  with torch.no_grad():
    logits = model_new(img)
  scores, tags_present = logits.squeeze().topk(10)
  tags = ""
  for ind in tags_present:
    tags += "#" + str(ind_to_tag[ind.item()]) + " "
  return tags

gr.Interface(
    fn=predict_hashtags,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(lines=10)
).launch(share=True)