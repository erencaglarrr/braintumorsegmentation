import os
import warnings
import pickle
warnings.filterwarnings('ignore')
import torch
from torch.utils.data import SubsetRandomSampler
import numpy as np
import bts.dataset as dataset
import bts.model as model
import bts.classifier as classifier
import bts.plot as plot
import matplotlib.pyplot as plt
import streamlit as st

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

st.title("BEYİN TÜMÖRÜ SEGMENTASYONU")
st.sidebar.image("indir.png")
selectbox = st.sidebar.selectbox(
    "Bir sayfa seçiniz",
    ("Anasayfa", "Segmentasyon", "İletişim")
)
st.sidebar.subheader("About App")
st.sidebar.text("-----------")
st.sidebar.text("@erencaglar")
st.sidebar.text("Since 2021")

#print('Computation Details')
#print(f'\tDevice Used: ({device})  {torch.cuda.get_device_name(torch.cuda.current_device())}\n')
#print('Packages Used Versions:-')
#print(f'\tPytorch Version: {torch.__version__}')


# To Start TensorBoard
# tensorboard --logdir logs --samples_per_plugin images=200
TEST_SPLIT = 0.2
BATCH_SIZE = 6
FILTER_LIST = [16,32,64,128,256]
MODEL_NAME = f"UNet-{FILTER_LIST}.pt"
DATASET_USED = 'png_dataset'
DATASET_PATH = os.path.join('dataset',DATASET_USED)

def get_indices(length, new=False):
    """ Gets the Training & Testing data indices for a
    paticular "DATASET_USED".Stores the indices and returns
    them back when the same dataset is used.
    Parameters:
        length(int): Length of the dataset used.
        new(bool): Discard the saved indices and get new ones.
    Return:
        train_indices(list): Array of indices used for training purpose.
        test_indices(list): Array of indices used for testing purpose.
    """
    # Pickle file location of the indices.
    file_path = os.path.join('dataset',f'split_indices_{DATASET_USED}.p')
    data = dict()
    if os.path.isfile(file_path) and not new:
        # File found.
        with open(file_path,'rb') as file :
            data = pickle.load(file)
            return data['train_indices'], data['test_indices']
    else:
        # File not found or fresh copy is required.
        indices = list(range(length))
        np.random.shuffle(indices)
        split = int(np.floor(TEST_SPLIT * len(tumor_dataset)))
        train_indices , test_indices = indices[split:], indices[:split]
        # Indices are saved with pickle.
        data['train_indices'] = train_indices
        data['test_indices'] = test_indices
        with open(file_path,'wb') as file:
            pickle.dump(data,file)
    return train_indices, test_indices

tumor_dataset = dataset.TumorDataset(DATASET_PATH)
train_indices, test_indices = get_indices(len(tumor_dataset))
train_sampler, test_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(test_indices)
trainloader = torch.utils.data.DataLoader(tumor_dataset, BATCH_SIZE, sampler=train_sampler)
testloader = torch.utils.data.DataLoader(tumor_dataset, 1, sampler=test_sampler)

unet_model = model.DynamicUNet(FILTER_LIST)
unet_classifier = classifier.BrainTumorClassifier(unet_model,device)
unet_classifier.restore_model(os.path.join('saved_models',MODEL_NAME))

if selectbox == "Anasayfa":
    st.title("Beyin tümörü nedir?")
    st.write("Beyin tümörü, beyin hücrelerinde doğal olmayan ve kontrolsüz büyüme olarak tanımlanabilir. İnsan kafatası katı ve hacim sınırlı bir vücut olduğu için, sonuç olarak, beklenmedik herhangi bir büyüme, beynin ilgili bölümüne göre bir insan işlevini etkileyebilir; dahası diğer vücut organlarına yayılabilir ve insan fonksiyonlarını etkileyebilir.")
    st.image("dataset/png_dataset/20.png")
    st.title("Segmentasyon nedir?")
    st.write("Olarak dijital görüntü işleme ve bilgisayar görme , görüntü bölümleme bir bölümleme işlemidir dijital görüntü , birden çok parçalı (halinde setleri arasında piksel aynı zamanda görüntü nesnelerinin olarak da bilinir). Segmentasyonun amacı, bir görüntünün temsilini basitleştirmek ve / veya daha anlamlı ve daha kolay analiz edilebilecek bir şeye dönüştürmektir. Görüntü bölütleme tipik olarak görüntülerdeki nesneleri ve sınırları (çizgiler, eğriler, vb.) Bulmak için kullanılır. Daha kesin olarak, görüntü bölümleme, bir görüntüdeki her piksele, aynı etikete sahip piksellerin belirli özellikleri paylaşacağı şekilde bir etiket atama işlemidir. Görüntü bölümlemenin sonucu, tüm görüntüyü toplu olarak kaplayan bir dizi bölüm veya görüntüden çıkarılan bir dizi konturdur (bkz. Kenar algılama ). Bir bölgedeki piksellerin her biri, renk , yoğunluk veya doku gibi bazı karakteristik veya hesaplanan özelliklere göre benzerdir . Bitişik bölgeler, aynı özellik(ler) açısından önemli ölçüde farklıdır. Tıbbi görüntülemede tipik olan bir görüntü yığınına uygulandığında , görüntü bölümlemesinden sonra ortaya çıkan konturlar, yürüyen küpler gibi enterpolasyon algoritmalarının yardımıyla 3D rekonstrüksiyonlar oluşturmak için kullanılabilir.")
    st.image("dataset/png_dataset/20_mask.png")

if selectbox == "Segmentasyon":
    i = st.text_input("Bir sayi giriniz: ")

    if st.button("Gönder"):
        # Run this cell repeatedly to see some results.
        image_index = test_indices[int(i)]
        sample = tumor_dataset[image_index]
        image, mask, output, d_score = unet_classifier.predict(sample,0.65)
        title = f'Name: {image_index}.png   Dice Score: {d_score:.5f}'
        # save_path = os.path.join('images',f'{d_score:.5f}_{image_index}.png')
        #plot.result(image,mask,output,title,save_path=None)
        st.header("Beyin MR")
        st.image(image)
        st.header("Maskelenmiş Tümör")
        st.image(mask)

        i = int(i)
        i += 1
        if i >= len(test_indices):
            i = 0


if selectbox == "İletişim":
    st.subheader("E-mail: caglarerennn@gmail.com")
    st.subheader("Telefon: 05355528331")
    st.subheader("Kocaeli Üniversitesi Umuttepe Yerleşkesi 41001, İzmit/KOCAELİ")

    import streamlit as st
    from streamlit_folium import folium_static
    import folium
    # center on Liberty Bell
    m = folium.Map(location=[40.821663, 29.925083], zoom_start=16)
    # add marker for Liberty Bell
    tooltip = "Kocaeli Üniversitesi"
    folium.Marker(
        [40.821663, 29.925083], popup="Kocaeli Üniversitesi", tooltip=tooltip
    ).add_to(m)
    # call to render Folium map in Streamlit
    folium_static(m)
