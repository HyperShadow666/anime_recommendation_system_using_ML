import pickle
import streamlit as st
import numpy as np

st.header("Anime recommendation system using Machine Learning")

model = pickle.load(open('artifacts/model.pkl','rb'))
anime_name = pickle.load(open('artifacts/anime_name.pkl','rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl','rb'))
anime_pivot = pickle.load(open('artifacts/anime_pivot.pkl','rb'))


selected_anime = st.selectbox("select or type your anime", anime_name)

def fetch_poster(suggestion):
    anime_name = []
    ids_index = []
    poster_url = []

    for anime_id in suggestion:
        anime_name.append(anime_pivot.index[anime_id])

    for name in anime_name[0]:
        ids = np.where(final_rating['title'] == name[0][0])
        ids_index.append(ids)

        url = final_rating.loc[final_rating['title'] == name, 'img_url'].iloc[0]
        poster_url.append(url)
    return poster_url

def recommend_anime(anime_name):
    anime_list = []

    #extract the recommendations

    anime_id = np.where(anime_pivot.index == anime_name)[0][0]
    distance, suggestion = model.kneighbors(anime_pivot.iloc[anime_id, :].values.reshape(1, -1), n_neighbors=11)

    #extract the posters
    poster_url = fetch_poster(suggestion)

    for i in range(len(suggestion)):
        animes = anime_pivot.index[suggestion[i]]
        for j in animes:
            anime_list.append(j)
    return anime_list, poster_url

if st.button('Show recommendation'):
    recommendation_animes, poster_url = recommend_anime(selected_anime)
    col1, col2, col3 = st.columns(3)
    col6, col7, col8 = st.columns(3)

    image_width = 200

    with col1:
        st.text(recommendation_animes[1])
        st.image(poster_url[1], width=image_width)

    with col2:
        st.text(recommendation_animes[2])
        st.image(poster_url[2], width=image_width)

    with col3:
        st.text(recommendation_animes[3])
        st.image(poster_url[3], width=image_width)


    with col6:
        st.text(recommendation_animes[6])
        st.image(poster_url[4], width=image_width)

    with col7:
        st.text(recommendation_animes[7])
        st.image(poster_url[5], width=image_width)

    with col8:
        st.text(recommendation_animes[8])
        st.image(poster_url[6], width=image_width)


