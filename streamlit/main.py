import streamlit as st
import pandas as pd
import time
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")

##### HEADER #####

st.title("Natural Language Processing")
st.subheader("First Year Project, Twitter NLP")
st.caption("*IT-University of Copenhagen, Bsc. in Data Science*")
st.caption("By Juraj Septak, Gusts Gustav, Franek Liszka, Mirka and Jannik Elsäßer *(Group E2)*")
st.write("------------------------------------------")

sidebar_options = (
    "Start Page", 
    "Preprocessing", 
    "Data Characterisation", 
    "Manual Annotation", 
    "Automatic Prediction", 
    "Data Augmentation")

melanoma_image = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Twitter-logo.svg/768px-Twitter-logo.svg.png"

##### PAGE CODE ##########
def start_page():
    st.sidebar.write("---------------------")
    st.sidebar.success("Start Page showing on the right:")
    
    col1, col2 = st.columns(2)

    with col1:
        st.write("""
        This interactive app is designed as a representation of our groups submission
        for the First Year Project 2, using different NLP Machine Learning Models, to classify different features.
        These project is focused on data scraped of twitter  
        On the left hand side you can choose different options from the sidebar.
        These are all different tasks of our project.  
        """)

    with col2:
        st.image(melanoma_image, caption='Put Twitter Word Cloud image Here', width=400)

    return

def preprocessing():
    st.sidebar.write("---------------------")
    st.sidebar.success("Page showing on the right:")

    st.write("This is where a breakdown of the algorithm, using an image from the dataset as an example, goes:")
    with st.expander("Masking and Segmenting the image:"):
        st.header("Loading image and putting mask on top")
        # downloading the images from the repo
        example_image = download_image(image_url, "example_image")
        example_mask = download_image(mask_url, "example_mask")

        algcol1, algcol2 = st.columns(2)
        with algcol1:
            st.write("The original Image")
            st.image(example_image)
        with algcol2:
            st.write("The mask")
            st.image(example_mask)
        st.write("The combined images:")
        test_mask = np.array(Image.open(example_mask))
        plot_image(test_mask)
        

def example_results_page():
    st.sidebar.write("---------------------")
    st.sidebar.success("Page showing on the right:")

    st.write("Not quite sure what we should put here, maybe results from our analysis, idk we'll figure it out.")

    return

def take_pic_page():
    st.sidebar.write("---------------------")
    st.sidebar.success("Page showing on the right:")

    take_picture = st.camera_input("Take a picture to test it for melanoma:")

    if take_picture:
        st.image(take_picture)
    
    st.write("Currently it just shows the picture, but here we can build the algorithm in and immediately show the results which would be really cool")
    return

def test_bulk_img():
    st.sidebar.write("---------------------")
    st.sidebar.success("Page showing on the right:")

    st.write("""
    The idea is that they can upload a folder with a bunch of images and test them on this page.
    This might be a bit difficult to implement. 
    """)
    return

############## FEATURE DETECTION CODE ###################

def masking(ima,test_mask):
    """
    Masking and cropping input image with binary mask
    """

    result = cv2.bitwise_and(ima,ima,mask=test_mask) #putting the mask on the image

    #cropping
    setm=set()
    setn=set()
    for x, i in enumerate(test_mask[:]):
        if 255 in i:
            setm.add(x)
    for i in range(test_mask.shape[1]):
        if 255 in test_mask[:,i]:
            setn.add(i)
    im2 = result[min(setm):max(setm),min(setn):max(setn),:]
    return im2
    
def segmenting(im2):
    """
    Divide the pixels into segments (segment = piece of continuous color in the image)
    """
    segments_slic = slic(im2, n_segments=100, compactness=10, sigma=1, start_label=1)
    return segments_slic

    # fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True, sharey=True)
    # ax[0].imshow(im2)
    # ax[0].set_title("Original")
    # ax[1].imshow(mark_boundaries(im2, segments_slic))
    # ax[1].set_title('SLIC')
    # for a in ax.ravel():
    #     a.set_axis_off()
    # plt.tight_layout()
    # plt.show()

###### DOWNLOADING IMAGE DATA CODE ###############

image_url = "https://github.com/jannik-el/melanoma-detection-app/blob/main/data/example-images/ISIC_0001769.jpg?raw=true"
mask_url = "https://github.com/jannik-el/melanoma-detection-app/blob/main/data/example-mask/ISIC_0001769_segmentation.png?raw=true"

def download_image(url, name):
    download = requests.get(url).content
    f = open(f'{name}.jpg', 'wb')
    f.write(download)
    f.close()
    return f"{name}.jpg"

######## OTHER BOILERPLATE CODE ##############

def plot_image(image):
    fig, ax = plt.subplots()
    ima=np.array(Image.open(image))
    ax.imshow(ima)
    return st.pyplot(fig)


###### MAIN FUNCTION #################

def main():

    st.sidebar.title("Explore the following:")
    st.sidebar.write("---------------------")
    app_mode = st.sidebar.selectbox("Please select from the following:", sidebar_options)

    if app_mode == "Start Page":
        start_page()

    elif app_mode == "Algorithm Description":
        preprocessing()

    elif app_mode == sidebar_options[2]:
        example_results_page()

    elif app_mode == sidebar_options[3]:
        take_pic_page()

    elif app_mode == sidebar_options[4]:
        test_bulk_img()

if __name__ == "__main__":
    main()