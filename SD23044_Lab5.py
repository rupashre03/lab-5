import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd


# Configure page settings
st.set_page_config(
    page_title="Computer Vision Image Classification",
    layout="centered"
)

# App title
st.title("Computer Vision Image Classification App")

# Short description
st.write("""
This web application demonstrates image classification using a
pre-trained deep learning model in PyTorch.
""")

# Configure device to CPU only
device = torch.device("cpu")

# Load pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Move model to CPU
model = model.to(device)

# Set model to evaluation mode
model.eval()


# Define image preprocessing transformations for ResNet18
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Step 6: Image upload interface
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)


    # -----------------------------
    # Step 7: Preprocess & inference
    # -----------------------------
    input_tensor = preprocess(image)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

    # -----------------------------
    # Step 8: Softmax & Top-5 predictions
    # -----------------------------

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Get top-5 probabilities and class indices
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    # Load ImageNet class labels
    labels = models.ResNet18_Weights.DEFAULT.meta["categories"]

    # Prepare results for display
    results = []
    for i in range(5):
        results.append({
            "Class": labels[top5_catid[i]],
            "Probability (%)": round(top5_prob[i].item() * 100, 2)
        })

    # Display results as a table
    st.subheader("Top 5 Predictions")
    st.table(pd.DataFrame(results))

    
    # -----------------------------
    # Step 9: Bar chart visualization
    # -----------------------------

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

    st.subheader("Prediction Probabilities (Top 5)")
    st.bar_chart(
        data=df_results.set_index("Class")["Probability (%)"]
    )

else:
    st.info("Please upload an image to begin classification.")
