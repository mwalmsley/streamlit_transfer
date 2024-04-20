
import streamlit as st
import numpy as np
import pandas as pd


import shared


def show_galaxies(galaxies, max_display_galaxies=18):
    
    galaxies['url'] = list(galaxies.apply(shared.get_url, axis=1))

    shared.show_galaxy_table(galaxies, max_display_galaxies)
    st.text(" \n")

    opening_html = '<div style=display:flex;flex-wrap:wrap>'
    closing_html = '</div>'
    child_html = ['<img src="{}" style=margin:3px;width:200px;></img>'.format(url) for url in galaxies['url'][:max_display_galaxies]]

    gallery_html = opening_html
    for child in child_html:
        gallery_html += child
    gallery_html += closing_html

    st.markdown(gallery_html, unsafe_allow_html=True)


def show_query_galaxy(galaxy):

    galaxy['url'] = shared.get_url(galaxy)
    
    st.header('Closest Galaxy')

    st.image(galaxy['url'], width=200)

    coords_string = 'RA: {:.5f}. Dec: {:.5f}'.format(galaxy['ra'], galaxy['dec'])
    viz_string = '[Search Vizier]({})'.format(shared.get_vizier_search_url(galaxy['ra'], galaxy['dec']))
    
    st.write(coords_string + '. ' + viz_string)


def main():

    st.title('Similarity Search')
    st.subheader('by Mike Walmsley ([@mike\_walmsley\_](https://twitter.com/mike_walmsley_))')
    st.text(" \n")

    ra = float(st.text_input('RA (deg)', key='ra', help='Right Ascension of galaxy to search (in degrees)', value='184.6750'))
    dec = float(st.text_input('Dec (deg)', key='dec', help='Declination of galaxy to search (in degrees)', value='11.73181'))

    legacysurvey_str_default = 'https://www.legacysurvey.org/viewer?ra=184.6750&dec=11.73181&layer=ls-dr8&zoom=12'
    legacysurvey_str = st.text_input('or, paste link', key='legacysurvey_str', help='Legacy Survey Viewer link (for ra and dec)', value=legacysurvey_str_default)
    
    if legacysurvey_str != legacysurvey_str_default:
        query_params = legacysurvey_str.split('?')[-1]
        ra_str = query_params.split('&')[0].replace('ra=', '')
        dec_str = query_params.split('&')[1].replace('dec=', '')
        ra = float(ra_str)
        dec = float(dec_str)
        st.markdown(f'Using link coordinates: {ra}, {dec}')

    with st.spinner('Loading representation, please wait'):
        # essentially all the delay
        # do this after rendering the inputs, so user has something to look at
        df, features = shared.prepare_data()
        print('data ready')
        go = st.button('Search')
        # st.markdown('Ready to search.')

    with st.expander('Important Notes'):
        st.markdown(
            """
            Which galaxies are included?
            - Galaxies must be between r-mag 14.0 and 19 (the SDSS spectroscopic limit).
            - Galaxies must be extended enough to be included in Galaxy Zoo (roughly, petrosian radius > 3 arcseconds)
            - Galaxies must be in the DECaLS DR8 sky area. A sky area chart will display if the target coordinates are far outside.
            
            What are the machine learning limitations?
            - The underlying model does not receive colour information to avoid bias. Colour grz images are shown for illustration only.
            - The underlying model is likely to perform better with "macro" morphology (e.g. disturbances, rings, etc.) than small anomalies in otherwise normal galaxies (e.g. supernovae, Voorwerpen, etc.)
            - Finding no similar galaxies does not imply there are none.
            
            Please see the paper (in prep.) for more details.
            """
        )
    st.text(" \n")



    # avoid doing a new search whenever ra OR dec changes, usually people would change both together
    if go:

        with st.spinner(f'Searching {len(df)} galaxies.'):
            
            coordinate_query = np.array([ra, dec]).reshape((1, -1))
            separation, best_index = shared.find_neighbours_from_query(df[['ra', 'dec']], coordinate_query)  # n_neigbours=1
            # print('crossmatched')

            shared.separation_warning(separation)
        
            neighbour_indices = shared.find_neighbours_from_index(features, best_index)
            assert neighbour_indices[0] == best_index  # should find itself

            query_galaxy = df.iloc[best_index]
            neighbours = df.iloc[neighbour_indices[1:]]

            # exclude galaxies very very close to the original
            # sometimes catalog will record one extended galaxy as multiple sources
            nontrivial_neighbours = shared.get_nontrivial_neighbours(query_galaxy, neighbours)

        show_query_galaxy(query_galaxy)
        
        st.header('Similar Galaxies')

        show_galaxies(nontrivial_neighbours, max_display_galaxies=18)


st.set_page_config(
    # layout="wide",
    page_title='DESI Active Search',
    page_icon='gz_icon.jpeg'
)



if __name__ == '__main__':

    # logging.basicConfig(level=logging.CRITICAL)


    # streamlit run similarity.py --server.fileWatcherType none


    # LOCAL = os.getcwd() == '/home/walml/repos/decals_similarity'
    # logging.info('Local: {}'.format(LOCAL))

    main()

