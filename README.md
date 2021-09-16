# SceneClassification
Dataset

This repo contains many models I used to classify scenes i.e. Outdoor vs Indoor and further classify outdoor scene into rain, snow, foggy, fog and normal.
The methods used for classifying Outdoor vs indoor are:

i. VGG16

ii. Resnet 18

iii. Resnet34

iv. Mobilenet v2
Models that were used for Places2.zip dataset that is primarily outdoor vs indoor

    Places2_VGG16.ipynb
    Places2_Resnet34.ipynb
    Places2_Resnet18(1st Approach).ipynb
Places2_Resnet18(2nd Approach).ipynb
Places2_MobilenetV2.ipynb

The files that can be used to test the real time data for Places2 dataset is
    CheckResnet34.py
    CheckResnet18.py
    CheckMobileNetV2.py

Models used to train wea.zip that has classification as fog, rain, snow, normal, night
    Resnet18_adv.ipynb
    MobilenetV2_aug.ipynb
    Kfold_resnet18.ipynb

The files that can be used to test the real time data for wea.zip  dataset is
    Res.py


Models used on the mixed dataset featureextract.zip dataset
    MobilenetV2_aug&SVM.ipynb
This model has a feature extraction model which extracts features of the indoor dataset on the featureextract.zip. After feature extraction we get 
    Testdata_svm(1).h5
    Traindata_svm(1).h5

These files are further trained using SVM classifier present in MobilenetV2_aug&SVM.ipynb for fog, rain, snow, normal, night classification

