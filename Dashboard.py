import streamlit as st 
import numpy as np
import nibabel as nib
import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
import torchio as tio
from PIL import Image
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os
import gdown
from nilearn import image

@st.cache_resource
def load_model(path = None):
    # If a copy of the model is present on the pc, load it from the path
    # Download it otherwise
    if path is None:
        id = '1B5pIMRFd0FywUdexZ5Ah6eoiI_9CfHag'
        path = 'model_best.pth'
        gdown.download(id=id, output=path, quiet=False)
    
    model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2
    )
    
    state = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    return model

#Initialize the image
def input_model(input_image):
    input_image = np.expand_dims(input_image, [0,3])
    input_image = st.session_state['transform'](input_image )
    input_image = np.squeeze(input_image)
    input_image = np.array(input_image)
    input_image = st.session_state['resizing'](input_image)     
    input_image = torch.unsqueeze(input_image, 0)

    return input_image

if __name__ == '__main__':
    
    # Set the tab bar
    gdown.download(id='1GagTCN-WJHUqdOKbQ79zIUjiISx0CYeZ', output='mri.png', quiet=True)
    icon = Image.open("mri.png")
    st.set_page_config(
        page_title="Spleen Detector",
        page_icon=icon,
        layout="wide",
    )

    #Set the Markdown
    st.markdown(
        """
        <style>
            [data-testid="stHeader"] {
                background-color: transparent;
            }
            [data-testid="stSidebar"]:first-child > div:first-child > div:nth-child(2) {
                margin-top: -6rem;
            }
            .main > .block-container {
                padding-top: 0;
            }
            #MsainMenu {
                visibility: hidden;
            }
            footer {
                visibility: hidden;
            }
            body {
                background-color: #a83236;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    #Setting the transformations
    st.session_state['transform'] = tio.Compose([
                                        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
                                        tio.Clamp(out_min=-1000, out_max=1400),
                                        tio.RescaleIntensity(out_min_max=(0, 1)),
                                        tio.Resample(1),
                                    ])
    st.session_state['resizing'] = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((224, 224), antialias=False),
                                    ])
    
  
    st.title(":red[Where is my Spleen?] :male-doctor:")
    st.markdown("")

    st.info("With this Dashboard is possible to upload an MRI image and create a mask for the identification of the spleen")
    
    #Loading the model
    gdown.download(id='1B5pIMRFd0FywUdexZ5Ah6eoiI_9CfHag', output='unet_model2d.pth', quiet=True)
    model = load_model("unet_model2d.pth")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown('---')
        st.header('1. Uploading Images')
        # Radio button
        type_file = st.radio("What kind of MRI file you want to upload?",
            ('Numpy', 'Nibabel'))

        
        # Upload the file if format is .npy
        if type_file == 'Numpy':
            
            st.header('Upload the MRI file')
            
            #Create file uploader for the numpy file
            file_npy = st.file_uploader("Choose an MRI numpy file", 
                                       accept_multiple_files = False,
                                       type = 'npy',
                                       help = 'Requires .npy file.')
            
            
            # Store it in session state
            if file_npy is not None:
                image = np.load(file_npy, allow_pickle=False)
                st.session_state['MRI_image'] = image
         
        # Upload the file if format is .nii.gz
        if type_file == 'Nibabel':
            
            st.header('Upload the MRI file')

            #loading the file's path
            path_nii = st.text_input('Input the path of the file you want to upload')
            if st.button("Load file"):
                if path_nii is not None:
                    path_nii = path_nii.replace(f"\\", "/")
                    path_nii = path_nii.replace(f"\"", "")
                    if not os.path.isfile(path_nii):
                        st.text('Do not exists any file under the input path')
                    
                    else:  
                        image = nib.load(path_nii).get_fdata()  
                        # Store it in session state
                        st.session_state['MRI_image'] = image
    

    with col2:
        
        if 'MRI_image' in st.session_state:
            if st.session_state['MRI_image'] is not None:

                with st.container():

                    st.markdown('---')
                    #Visualize the chosen slice
                    st.header('2. Visualize Slice')
                    if len(np.shape(st.session_state['MRI_image'])) == 3:
                        #if image is 3D then apply a slicer to select the preferred slice
                        dim = st.slider('Select the slice to visualize?', 0, np.shape(st.session_state['MRI_image'])[2], np.shape(st.session_state['MRI_image'])[2]//2)
                        fig, ax = plt.subplots(figsize=(3, 3))
                        ax.set_axis_off()
                        ax.imshow(st.session_state['MRI_image'][:,:,dim], cmap = "gray")
                        st.pyplot(fig)

                    #Plotting 2D image
                    if len(np.shape(st.session_state['MRI_image'])) == 2:
                        fig, ax = plt.subplots(figsize=(3, 3))
                        ax.set_axis_off()
                        ax.imshow(st.session_state['MRI_image'], cmap = "gray")
                        st.pyplot(fig, use_container_width = False) 

                with st.container():
                    st.markdown('---')
                    st.header('3. Predict the mask')
                    #Creating the mask
                    if st.button("Predict"):
                        #Cleaning the previously saved images
                        st.session_state['pred_mask'] = None
                        st.session_state['mask_3d'] = None

                        #For 3D images:
                        if len(np.shape(st.session_state['MRI_image'])) == 3:
                            #Creating an image and a mask with only zeros
                            image_3d = np.zeros([224, 224, np.shape(st.session_state['MRI_image'])[2]])
                            mask_3d = np.zeros([224, 224, np.shape(st.session_state['MRI_image'])[2]])
                            for i in range(np.shape(st.session_state['MRI_image'])[2]):
                                input_image = input_model(st.session_state['MRI_image'][:,:,i])

                                #Creating the predicted mask     
                                model_output = model(input_image.float())
                                pred_mask = torch.where(F.softmax(model_output, dim=1)[:, 1] >= 0.5, 1, 0)
                                mask_3d[:,:,i] = pred_mask[0]
                                image_3d[:,:,i] = input_image[0,0,:,:]
                            #Saving in session state the mask and the image
                            st.session_state['image_3d'] = image_3d
                            st.session_state['mask_3d'] = mask_3d
                        
                        #For 2D images:
                        if len(np.shape(st.session_state['MRI_image'])) == 2:
                            #Apply the transformations
                            input_image = input_model(st.session_state['MRI_image'])
                            model_output = model(input_image.float())
                            st.session_state['pred_mask'] = torch.where(F.softmax(model_output, dim=1)[:, 1] >= 0.5, 1, 0)
                            st.session_state['image_2d'] = input_image

                            
                    #Plotting the 3D mask
                    if 'mask_3d' in st.session_state:
                        if st.session_state['mask_3d'] is not None:
                            #using a slicer to select the preferred slice
                            dim = st.slider('Select the slice to visualize', 0, np.shape(st.session_state['image_3d'])[2], np.shape(st.session_state['image_3d'])[2]//2)
                            
                            fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12, 6))

                            ax1.imshow(st.session_state['image_3d'][:,:,dim], cmap = "gray")
                            ax2.imshow(st.session_state['mask_3d'][:,:,dim], cmap = "rainbow")
                            ax3.imshow(st.session_state['image_3d'][:,:,dim], cmap = "gray")
                            ax3.imshow(st.session_state['mask_3d'][:,:,dim], cmap = "rainbow", alpha = 0.5)

                            ax1.set_title("MRI")
                            ax2.set_title("PREDICTED MASK")
                            ax3.set_title("MRI w/ MASK")

                            ax1.set_axis_off()
                            ax2.set_axis_off()
                            ax3.set_axis_off()
                            st.pyplot(fig)

                    #Plotting 2D predicted mask
                    if 'pred_mask' in st.session_state:
                        if st.session_state['pred_mask'] is not None:

                            fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12, 6))

                            input_image = torch.squeeze(st.session_state['image_2d'], dim = 0)

                            ax1.imshow(input_image[0], cmap = "gray")
                            ax2.imshow(st.session_state['pred_mask'][0], cmap = "rainbow")
                            ax3.imshow(input_image[0], cmap = "gray")
                            ax3.imshow(st.session_state['pred_mask'][0], cmap = "rainbow", alpha = 0.5)

                            ax1.set_title("MRI")
                            ax2.set_title("PREDICTED MASK")
                            ax3.set_title("MRI w/ MASK")

                            ax1.set_axis_off()
                            ax2.set_axis_off()
                            ax3.set_axis_off()
                            st.pyplot(fig)
                    