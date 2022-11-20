import numpy as np
import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


@st.cache(allow_output_mutation=True)
def load_model():
    #   model=tf.keras.models.load_model('/content/image_classification.hdf5')
    net = models.vgg19_bn(pretrained=False)
    # 最終ノードの出力を2にする
    in_features = net.classifier[6].in_features
    net.classifier[6] = nn.Linear(in_features, 2)
    net.load_state_dict(torch.load("PizzaNet.pkl", map_location=torch.device("cpu")))
    return net


with st.spinner("Model is being loaded.."):
    net = load_model()

st.write(
    """
         # 🤖Pizza Net🍕
         """
)

file = st.file_uploader(
    "Upload the image to be classified❗", type=["jpg", "png", "jpeg"]
)
st.set_option("deprecation.showfileUploaderEncoding", False)


def upload_predict(upload_image, model):

    # size = (180,180)
    # img = Image.open(upload_image)
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )
    img = val_transform(upload_image)
    model.eval()
    prediction = model.forward(img.unsqueeze(0))
    # predicted_class = np.argmax(prediction).item()

    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = upload_predict(image, net)
    image_class = np.argmax(predictions.detach().numpy())
    if image_class == 1:
        ans = "This is a Pizza❗"
    else:
        ans = "This is Not a Pizza❗"

    softmax = nn.Softmax()
    score = np.round(torch.max(softmax(predictions)).item()) * 100
    st.write(
        """
             # Result(判定結果): """,
        ans,
    )
    st.write(
        """
             # Score(%): """,
        score,
        "%",
    )
    print("AIの判定結果 ", image_class, "AIの確信度(%)", score, "%")
