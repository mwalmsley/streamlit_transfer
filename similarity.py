
import streamlit as st
import numpy as np
import pandas as pd

import shared
import learning


def show_galaxies(df, indices, max_display_galaxies=6):

    galaxies = df.loc[indices]
    galaxies = galaxies[:max_display_galaxies]
    
    galaxies['url'] = list(galaxies.apply(shared.get_url, axis=1))

    st.header('Similar Galaxies')

    toggles = []
    n_cols = 3
    n_rows = len(galaxies) // n_cols
    with st.form(key='label_form', clear_on_submit=True):
        for row_i in range(0, n_rows):
            row_columns = st.columns(n_cols)
            for col_i in range(n_cols):
                image_n = (row_i*n_cols) + col_i
                galaxy_index = galaxies.index[image_n]
                if n_cols*row_i+col_i < len(galaxies):
                    col = row_columns[col_i]
                    col.image(galaxies.loc[galaxy_index, 'url'])
                    toggles.append(col.toggle(f'test_{galaxy_index}', False))
        submitted = st.form_submit_button('Submit labels') # , on_click=add_labels_to_session_state, args=(indices, toggles))

    if submitted:
        add_labels_to_session_state(indices, toggles)
        st.rerun()


def add_labels_to_session_state(indices, toggles):
    print('record positions of all toggles')
    for galaxy_index, toggle in zip(indices, toggles):
        st.session_state['labels'][int(galaxy_index)] = int(toggle)
    print(st.session_state['labels'])

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
    st.subheader('by [Mike Walmsley](walmsley.dev)')
    st.text("\n")

    st.info(st.session_state.get('labels', {}))
    
    shared.add_important_notes_expander()

    ra, dec = user_coordinate_input()

    with st.spinner('Loading representation, please wait'):
        # essentially all the delay
        # do this after rendering the inputs, so user has something to look at
        df, _ = shared.prepare_data()
        # display when ready
        go = st.button('Cross-match')
        
    # avoid doing a new search whenever ra OR dec changes, usually people would change both together
    if go:

        with st.spinner(f'Cross-matching galaxy'):

            st.session_state['labels'] = {}  # wipe labels, start fresh
            
            coordinate_query = np.array([ra, dec]).reshape((1, -1))
            separation, best_index = shared.find_neighbours_from_query(df[['ra', 'dec']], coordinate_query)  # n_neigbours=1

            shared.separation_warning(separation)

            query_galaxy = df.iloc[best_index]

            show_query_galaxy(query_galaxy)

            # wipe label state and set this galaxy (only) as true label
            st.session_state['labels'][int(best_index)] = 1


    st.header('Galaxies to label')

    # load current labels
    df['has_label'] = False
    df['label'] = np.nan
    # print(df.loc[1478084])
    for label_index, label in st.session_state.get('labels', {}).items():
        df.loc[label_index, 'has_label'] = True
        df.loc[label_index, 'label'] = label

    query_indices, learner = learning.run_active_learning_iteration(batch_size=6, df=df, tree=None)

    st.button('Show galaxies to label')

    # show_galaxies(df, [0, 1, 2, 3, 4, 5], max_display_galaxies=6)
    show_galaxies(df, query_indices, max_display_galaxies=6)

def user_coordinate_input():
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
    return ra,dec


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

