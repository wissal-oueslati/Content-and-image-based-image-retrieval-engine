import streamlit as st
from elasticsearch import Elasticsearch
from PIL import Image
import io
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import base64
import time

# Load the pre-trained ResNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Modify the resnet to exclude the final FC layer
resnet = models.resnet50(pretrained=True).to(device)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))  # Remove the last FC layer
resnet.eval()


def search_by_feature_vector(vector):
    body = {
        "knn": {
            "field": "image_vector",
            "query_vector": vector,
            "k": 10,
            "num_candidates": 100
        },
        "_source": ["image_id"]
    }
    response = es.search(index='image_vector_index_final_version', body=body)
    return response


# Define a function to extract features
def extract_features(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(image)
    return features.cpu().squeeze().numpy()


# Create an Elasticsearch client
es = Elasticsearch("http://localhost:9200")

# Définissez le chemin de votre image de fond
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://th.bing.com/th/id/OIP.PmTTuduUWyaSYuRJGqxSzwHaE8?pid=ImgDet&rs=1");
background-size: cover;

}
[data-testid="stHeader"]{
background-color: rgba(0, 0, 0, 0);
}

</style>

"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Define the Streamlit app
st.markdown(
    f"""
    <h1 style='text-align: center; color: black;'>Search Engine</h1>
    """,
    unsafe_allow_html=True
)

# Add a selector to choose between textual search and image search
search_option = st.selectbox("Choose the search type", ["Search by text", "Search by image"])

if search_option == "Search by text":
    # Input text query from the user
    query = st.text_input("Enter a text query", "", key="text_query_input")

    # Perform a fuzzy search on Elasticsearch
    # Add a "Search" button
    if st.button("Search"):
        start_time = time.time()
        # Perform a fuzzy search on Elasticsearch
        if query:
            result = es.search(
                index='flickrdata',
                body={
                    'query': {
                        'fuzzy': {
                            'tags': query
                        }
                    }
                }
            )

            # Display the search results
            st.header("Search Results:")
            cols = st.columns(2)
            col_heights = [0, 0]
            for hit in result['hits']['hits']:
                image_url = f"http://farm{hit['_source']['flickr_farm']}.staticflickr.com/{hit['_source']['flickr_server']}/{hit['_source']['id']}_{hit['_source']['flickr_secret']}.jpg"

                col_id = 0 if col_heights[0] <= col_heights[1] else 1
                cols[col_id].image(image_url)
                col_heights[col_id] += 1

            end_time = time.time()

            st.success(
                f"Found {len(result['hits']['hits'])} images matching your query. Time taken: {end_time - start_time:.2f} seconds")

elif search_option == "Search by image":
    # Input for uploading an image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"],
                                     key="image_input")

    # Add a "Search" button for image search
    search_image_button = st.button("Search")

    # Image-based search
    if uploaded_file and search_image_button:
        start_time = time.time()
        # Extract features from the uploaded image
        image_stream = io.BytesIO(uploaded_file.getvalue())
        features = extract_features(image_stream)

        # Query Elasticsearch with image features
        result_vector_search = search_by_feature_vector(features)
        end_time = time.time()

        # Get the image IDs from the result
        image_ids = [hit['_source']['image_id'] for hit in result_vector_search['hits']['hits']]

        # Query the "flicker" index to get image information using these IDs
        body = {
            "query": {
                "terms": {
                    "id": image_ids
                }
            }
        }
        result_flicker = es.search(index='flickrdata', body=body)

        # Display the search results
        st.header("Search Results:")
        cols = st.columns(2)
        col_heights = [0, 0]
        for hit in result_flicker['hits']['hits']:
            image_url = f"http://farm{hit['_source']['flickr_farm']}.staticflickr.com/{hit['_source']['flickr_server']}/{hit['_source']['id']}_{hit['_source']['flickr_secret']}.jpg"
            col_id = 0 if col_heights[0] <= col_heights[1] else 1
            cols[col_id].image(image_url)
            col_heights[col_id] += 1

        st.success(
            f"Found {len(result_flicker['hits']['hits'])} images matching your query. Time taken: {end_time - start_time:.2f} seconds")

# Display some instructions
else:
    st.write("Choose a search type (Text Search or Image Search) to search for images in the Elasticsearch index.")


# Ajouter le footer avec la largeur à 100%
st.markdown(
    """
    <footer style='text-align: center; color: white; padding: 10px; background-color: black; position: fixed; bottom: 0; width: 100%; left: 0;'>
        Copyright © 2023 - Mariem Mezghani & Wissal Oueslati - AIM
    </footer>
    """,
    unsafe_allow_html=True
)
