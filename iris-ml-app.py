import streamlit as st
from PIL import Image
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import altair as alt
image = Image.open('logoo.jpg')
cola, colb, colc = st.columns([3,6,1])
with cola:
    st.write("")

with colb:
    st.image(image, width = 300)

with colc:
    st.write("")
menu = ["Home","About"]
choice = st.sidebar.selectbox("Menu",menu)
if choice == "Home":
    st.write("""
    # Simple Iris Flower Prediction App

    This app predicts the **Iris flower** type!
    """)
    cola, colb, colc = st.columns([1,6,1])
    irisi = Image.open('iris.jpg')
    with cola:
        st.write("")
    with colb:
        st.image(irisi, width = 600)
    with colc:
        st.write("")
    st.sidebar.header('User Input Parameters')

    def user_input_features():
        sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.21)
        sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 2.42)
        petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 6.71)
        petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.47)
        data = {'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width}
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_features()

    st.subheader('User Input Default Parameters')
    st.write("""
    You can change these parameters in the sidebar.
    """)
    st.write(df)

    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    clf = RandomForestClassifier()
    clf.fit(X, Y)

    prediction = clf.predict(df)
    prediction_proba = clf.predict_proba(df)


    # st.write(iris.target_names.transpose())



    cola, colb, colc = st.columns([4,6,1])
    Setosa = Image.open('Setosa.jpg')
    Virginica = Image.open('Virginica.jpg')
    Versicolor = Image.open('Versicolor.jpg')
    with cola:
        st.subheader('Lables and Indices')
        st.write("")
        st.write("""Class labels and their corresponding index number""")
        st.write(iris.target_names)
    with colb:
        st.subheader('Prediction')
        if (iris.target_names[prediction][0]=="setosa"):
            st.image(Setosa, width = 200)
            st.write("Setosa")
        elif (iris.target_names[prediction][0]=="virginica"):
            st.image(Virginica, width = 200)
            st.write("Virginica")
        elif (iris.target_names[prediction][0]=="versicolor"):
            st.image(Versicolor, width = 200)
            st.write("Versicolor")
    with colc:
        st.write("")
    #st.write(prediction)

    st.subheader('Prediction Probabilities')
    # st.write(prediction_proba)
    proba_df_clean = prediction_proba.T
    proba_df= pd.DataFrame(proba_df_clean, columns=["Probabilities"])
    flowers= ["Setosa","Versicolor","Virginica"]
    proba_df["Flowers"]= flowers
    # st.write(type(proba_df))
    column_names = ["Flowers", "Probabilities"]
    proba_df = proba_df.reindex(columns=column_names)
    st.write(proba_df)
    fig = alt.Chart(proba_df).mark_bar().encode(x='Flowers',y='Probabilities',color='Flowers')
    st.altair_chart(fig,use_container_width=True)
else:
    st.subheader("About")
    st.write("With a hybrid profile of data science and computer science, I’m pursuing a career in AI-driven firms. I believe in dedication, discipline, and creativity towards my job, which will be helpful in meeting your firm's requirements as well as my personal development.")
    st.write("Check out this project's [Github](https://github.com/bashirsadat/iris-ml)")
    st.write(" My [Linkedin](https://www.linkedin.com/in/saadaat/)")
    st.write("See my other projects [LinkTree](https://linktr.ee/saadaat)")
