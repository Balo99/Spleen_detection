# Spleen_detection
Exam Project in which there is an implementation of an AI that is able to detect the spleen given some MRI scans

The task have been chosen starting from the more famous Decathlon Challenge, but in my case I trained the model using only the spleen dataset. 
There are two models that can be used, the first is trained on 2D files where each files correspond to an MRI Slice and the second uses all the slices in a 3D version.

Finally the file called Dashboard is a fully functioning dashboard implemented in streamlit in which by inserting a numpy file is possible to obtain the computed mask
