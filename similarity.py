
import streamlit as st
import numpy as np
import pandas as pd


import shared
import learning


def show_galaxies(galaxies, max_display_galaxies=6):

    galaxies = galaxies[:max_display_galaxies]
    
    galaxies['url'] = list(galaxies.apply(shared.get_url, axis=1))

    st.header('Similar Galaxies')

    toggles = []
    with st.form(key='label_form', clear_on_submit=True):
        n_cols = 3
        n_rows = len(galaxies) // n_cols + 1
        for row_i in range(0, n_rows):
            row_columns = st.columns(n_cols)
            for col_i in range(n_cols):
                galaxy_n = n_cols*row_i+col_i
                if n_cols*row_i+col_i < len(galaxies):
                    col = row_columns[col_i]
                    col.image(galaxies.iloc[galaxy_n]['url'],)
                    toggles.append(col.toggle(f'test_{galaxy_n}', False))
        submitted = st.form_submit_button('Submit labels')
        if submitted:
            st.write(toggles)


    # shared.show_galaxy_table(galaxies, max_display_galaxies)
    # st.text(" \n")

    # opening_html = '<div style=display:flex;flex-wrap:wrap>'
    # closing_html = '</div>'
    # child_html = ['<img src="{}" style=margin:3px;width:200px;></img>'.format(url) for url in galaxies['url'][:max_display_galaxies]]

    # gallery_html = opening_html
    # for child in child_html:
    #     gallery_html += child
    # gallery_html += closing_html

    # st.markdown(gallery_html, unsafe_allow_html=True)


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
        go = st.button('Cross-match')
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

        with st.spinner(f'Cross-matching galaxy'):
            
            coordinate_query = np.array([ra, dec]).reshape((1, -1))
            separation, best_index = shared.find_neighbours_from_query(df[['ra', 'dec']], coordinate_query)  # n_neigbours=1

            shared.separation_warning(separation)

            query_galaxy = df.iloc[best_index]

            show_query_galaxy(query_galaxy)

            # wipe label state and set this galaxy (only) as true label
            st.session_state['labels'] = [(best_index, 1)]
            
        st.header('Galaxies to label')

        st.button('Show galaxies to label')

        show_galaxies(df, max_display_galaxies=6)

        # df.loc[best_index, 'has_label'] = True
        # df.loc[best_index, 'label'] = 1



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

