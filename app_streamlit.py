import streamlit as st
import pandas as pd
from sklearn.base import BaseEstimator
import pickle
from sklearn.metrics import mean_absolute_error
import requests
from streamlit_option_menu import option_menu
import altair as alt
import datetime
from io import BytesIO
import matplotlib.pyplot as plt
st.set_page_config(layout="wide", page_title='ALVA x Purwadhika Workshop')

class NoTransformer(BaseEstimator):
    """Passes through data without any change and is compatible with ColumnTransformer class"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X

# Load your model
@st.cache_resource
def load_model():
    github_model_url = 'https://raw.githubusercontent.com/OLR-Olrange/Purwadhika-Workshop/main/Final_Model_Catboost_v2.sav'
    response = requests.get(github_model_url)
    if response.status_code == 200:
        model = pickle.loads(response.content)
        return model
    else:
        return None
model_dict = load_model()

# Load data
@st.cache_data
def load_data(url: str):
    df = pd.read_csv(url)
    return df
df_residential = load_data('https://raw.githubusercontent.com/OLR-Olrange/Purwadhika-Workshop/main/DF_Train_Clean.csv')

# Function to calculate MAE
def count_mae():
    df_train = load_data('https://raw.githubusercontent.com/OLR-Olrange/Purwadhika-Workshop/main/DF_Train_Clean.csv')
    X_train = df_train.drop('PRICE', axis=1)
    y_train = df_train['PRICE']
    y_pred = model_dict['model'].predict(model_dict['transformer'].transform(X_train))
    mae = mean_absolute_error(y_train, y_pred)
    return mae

MAE = count_mae()

params = ['selected_ward', 'selected_quadrant', 'latitude', 'longitude', 'building_age', 'renovation_years', 'selected_saleyear', \
        'condition', 'grade', 'qualified', 'gba', 'landarea', 'selected_style', 'selected_usecode', 'rooms', 'bedrooms', 'bathrooms', \
        'half_bathrooms', 'kitchens', 'fireplaces', 'ac', 'selected_heat', 'roof', 'struct']

if any(param not in st.session_state for param in params):
    for param in params:
        st.session_state[param] = None
if 'predict_info' not in st.session_state:
    st.session_state.predict_info = None
if 'mae' not in st.session_state:
    st.session_state.mae = None
if 'input_data' not in st.session_state:
    st.session_state.input_data = None
if 'property_info' not in st.session_state:
    st.session_state.property_info = None
if 'submit' not in st.session_state:
    st.session_state.submit = False
if 'estimate_button' not in st.session_state:
    st.session_state.estimate_button = False

def clicked():
    st.session_state.submit = True

def predict_property_value():
    
    st.title('Property Value Estimator :house:')
    st.subheader('Fill the form to estimate property value!')
    st.caption('The required data to estimate the value of a property are the location, the condition, and the specification of the property.')
    st.divider()

    # Define unique values for dropdowns
    ward = sorted(df_residential['WARD'].unique().tolist())
    quadrant = df_residential['QUADRANT'].unique().tolist()
    heat = df_residential['HEAT'].unique().tolist()
    style = sorted(df_residential['STYLE'].unique().tolist())
    usecode = sorted(df_residential['USECODE'].unique().tolist())
    saleyear = sorted(df_residential['SALEYEAR'].unique().tolist(), reverse=True)

    # Create mapping for dropdowns
    condition_mapping = {'Poor': 0, 'Good': 1}
    grade_mapping = {'Fair quality': 0, 'Average': 1, 'Above average': 2, 'Good quality': 3, 'Very good': 4, 'Excellent': 5, 'Superior': 6}
    usecode_description = [
        '11 - Single family residential homes used as such',
        '12 - Single family residential home with non-economic 2nd unit',
        '13 - Single family residential home with slight commercial/ind',
        '15 - Townhouse - Planned Development',
        '19 - SFR - Manufactured Home',
        '23 - Triplex, double or duplex with single family home',
        '24 - Four living units; e.g. fourplex or triplex with SFR'
    ]
    usecode_mapping = dict(zip(usecode_description, usecode))
    roof_mapping = {
        'Concrete / Comp Shingle / Built Up / Metal-Pre / Typical / Composition Ro' : 0,
        'Shake / Metal-Sms / Shingle / Concrete Tile / Water Proof / Clay Tile / Slate / Neopren / Wood-FS / Metal-Cpr' : 1
    }
    struct_mapping = {
        'Semi-Detached / Multi / Town Inside / Town End / Row End' : 0,
        'Single / Row Inside / Default': 1
    }

    col1, col2 = st.columns(2)

    with st.form('predict_value', clear_on_submit=False, border=True):
        # Location
        st.session_state.selected_ward = col1.selectbox('Area', ward)
        st.session_state.selected_quadrant = col2.selectbox('Quadrant', quadrant)
        st.session_state.latitude = col1.number_input('Latitude', min_value=-90.0, max_value=90.0, step=0.01)
        st.session_state.longitude = col2.number_input('Longitude', min_value=-180.0, max_value=180.0, step=0.01)

        # Condition
        st.session_state.building_age = col1.number_input('Building Age (in years)', min_value=0, max_value=200, step=1)
        st.session_state.renovation_years = col2.number_input('Renovation Years (in years)', min_value=0, max_value=100, step=1)
        st.session_state.selected_saleyear = col1.selectbox('Last Sale Year', saleyear)
        st.session_state.condition = col2.selectbox('Condition', ['Poor', 'Good'])  
        st.session_state.grade = col1.selectbox('Grade', ['Fair quality', 'Average', 'Above average', 'Good quality', 'Very good', 'Excellent', 'Superior'])    
        st.session_state.qualified = col2.selectbox('Qualified', ['Yes', 'No'])

        # Specification
        st.session_state.gba = col1.number_input('Gross Building Area (in sqft)', min_value=0, max_value=10000, step=1)
        st.session_state.landarea = col2.number_input('Land Area (in sqft)', min_value=0, max_value=100000, step=1)
        st.session_state.selected_style = col1.selectbox('Style', style)
        st.session_state.selected_usecode = col2.selectbox('Use Code', usecode_description)
        st.session_state.rooms = col1.number_input('Rooms', min_value=0, max_value=20, step=1)
        st.session_state.bedrooms = col2.number_input('Bedrooms', min_value=0, max_value=10, step=1)
        st.session_state.bathrooms = col1.number_input('Bathrooms', min_value=0, max_value=10, step=1)
        st.session_state.half_bathrooms = col2.number_input('Half Bathrooms', min_value=0, max_value=5, step=1)
        st.session_state.kitchens = col1.number_input('Kitchens', min_value=0, max_value=5, step=1)
        st.session_state.fireplaces = col2.number_input('Fireplaces', min_value=0, max_value=5, step=1)
        st.session_state.ac = col1.selectbox('Air Conditioning', ['Yes', 'No'])
        st.session_state.selected_heat = col2.selectbox('Heat', heat)
        st.session_state.roof = col1.selectbox('Roof', ['Concrete / Comp Shingle / Built Up / Metal-Pre / Typical / Composition Ro', 'Shake / Metal-Sms / Shingle / Concrete Tile / Water Proof / Clay Tile / Slate / Neopren / Wood-FS / Metal-Cpr'])    
        st.session_state.struct = col2.selectbox('Structure', ['Semi-Detached / Multi / Town Inside / Town End / Row End', 'Single / Row Inside / Default'])  

        numeric_condition = condition_mapping[st.session_state.condition]
        numeric_grade = grade_mapping[st.session_state.grade]
        numberic_usecode = usecode_mapping[st.session_state.selected_usecode]
        numberic_roof = roof_mapping[st.session_state.roof]
        numberic_struct = struct_mapping[st.session_state.struct]

        if st.form_submit_button("Estimate"):
            st.session_state.estimate_button = True
            st.session_state.input_data = {
                'BATHRM': [st.session_state.bathrooms],
                'HF_BATHRM': [st.session_state.half_bathrooms],
                'ROOMS': [st.session_state.rooms],
                'BEDRM': [st.session_state.bedrooms],
                'ayb_age': [st.session_state.building_age],
                'eyb_age': [st.session_state.renovation_years],
                'GBA': [st.session_state.gba],
                'KITCHENS': [st.session_state.kitchens],
                'FIREPLACES': [st.session_state.fireplaces],
                'LANDAREA': [st.session_state.landarea],
                'LATITUDE': [st.session_state.latitude],
                'LONGITUDE': [st.session_state.longitude],
                'AC': [st.session_state.ac],
                'QUALIFIED': [st.session_state.qualified],
                'WARD': [st.session_state.selected_ward],
                'QUADRANT': [st.session_state.selected_quadrant],
                'HEAT': [st.session_state.selected_heat],
                'STYLE': [st.session_state.selected_style],
                'USECODE': [numberic_usecode],
                'STRUCT': [numberic_struct],
                'GRADE': [numeric_grade],
                'CNDTN': [numeric_condition],
                'ROOF': [numberic_roof],
                'SALEYEAR': [st.session_state.selected_saleyear]
            }

            df_predict = pd.DataFrame(st.session_state.input_data)

            # Transform and predict
            transformed_data = model_dict['transformer'].transform(df_predict)
            prediction = model_dict['model'].predict(transformed_data)[0]
            st.session_state.predict_info = "\\${:,.2f}".format(prediction)
            st.session_state.mae = "±\\${:,.2f}".format(MAE)
    
    if st.session_state.estimate_button:
        st.info(f'The estimated property value is **{st.session_state.predict_info}**')

        if st.button('Click here for property information!'):
            st.session_state.property_info = True
        
    if st.session_state.property_info and st.session_state.input_data is not None:        
        # if st.session_state.property_info == True:
        with st.container(border=True):
            st.header('Property information', divider=True)
            st.markdown(f'''
            **Value Estimation**:
            - Estimated Price: {st.session_state.predict_info}
            - Estimated Error (Mean Absolute Error): {st.session_state.mae}

            **Location**:
            - Area: {st.session_state.input_data['WARD'][0]}
            - Quadrant: {st.session_state.input_data['QUADRANT'][0]}
            - Longitude: {st.session_state.input_data['LONGITUDE'][0]}°
            - Latitude: {st.session_state.input_data['LATITUDE'][0]}°
            ''')

def data_viz():
    st.title('Data Visualization :bar_chart:')

    opt1, opt2 = st.columns(2)
    start_date = opt1.date_input("Select Start Date", datetime.date(2023, 1, 1))
    end_date = opt2.date_input("Select End Date")        

    content_data = load_data('https://raw.githubusercontent.com/OLR-Olrange/Purwadhika-Workshop/main/purwadhika_content.csv')
    content_data['post_time'] = pd.to_datetime(content_data['post_time'])
    content_data['engagement'] = content_data['likes_count'] + content_data['comments_count']

    channel_data = load_data('https://raw.githubusercontent.com/OLR-Olrange/Purwadhika-Workshop/main/purwadhika_channel.csv')
    channel_data['date'] = pd.to_datetime(channel_data['date'])

    if st.button("Apply"):
        content_data_slicing = content_data[(content_data['post_time'] >= f'{start_date}') & (content_data['post_time'] <= f'{end_date}')].sort_values('post_time').reset_index(drop=True)
        channel_data_slicing = channel_data[(channel_data['date'] >= f'{start_date}') & (channel_data['date'] <= f'{end_date}')].sort_values('date').reset_index(drop=True)

        content_data_slicing_copy = content_data_slicing.copy()
        content_data_slicing_copy['post_time'] = pd.to_datetime(content_data_slicing_copy['post_time']).dt.strftime('%Y-%m-%d')

        channel_data_slicing_copy = channel_data_slicing.copy()
        channel_data_slicing_copy['date'] = pd.to_datetime(channel_data_slicing_copy['date']).dt.strftime('%Y-%m-%d')

        df_groupby_type_mean = content_data_slicing_copy[['engagement', 'content_type']].groupby('content_type').mean()
        df_groupby = content_data_slicing_copy.groupby('content_type')['post_url'].nunique().reset_index(name='post_count')

        table1 , table2 = st.columns(2)
        table1.subheader("Content Performance")
        table1.dataframe(content_data_slicing_copy)

        table2.subheader("Channel Performance")
        table2.dataframe(channel_data_slicing_copy)

        st.markdown("---")
        with st.expander("View Charts"):
            st.subheader('Channel Growth')

            st.line_chart(channel_data_slicing_copy,x='date', y='followers_growth')

            st.subheader('Content Performance Overview')
            chartOne = alt.Chart(df_groupby_type_mean.reset_index()).mark_bar().encode(
                    x=alt.X('content_type:O', axis=alt.Axis(labelAngle=-0)),
                    y='engagement:Q',
                    color='content_type',
                    ).interactive()
            
            chartTwo = alt.Chart(df_groupby.reset_index()).mark_bar().encode(
                    x=alt.X('content_type:O', axis=alt.Axis(labelAngle=-0)),
                    y='post_count:Q',
                    color='content_type',
                    ).interactive()

            chart1 , chart2 = st.columns(2)        
            chart1.write(f"<h4 style='font-size: 16px; font-family: Arial, sans-serif;'>Average Engagement per Post Type</h4>", unsafe_allow_html=True)
            chart1.altair_chart(chartOne,theme=None,use_container_width=True)

            chart2.write(f"<h4 style='font-size: 16px; font-family: Arial, sans-serif;'>Number of Post per Post Type</h4>", unsafe_allow_html=True)
            chart2.altair_chart(chartTwo,theme=None,use_container_width=True)
            
            df_filtered = content_data_slicing_copy[(content_data_slicing_copy['likes_count'] > 0)]
            df_filtered['post_time'] = pd.to_datetime(df_filtered['post_time'], errors='coerce')
            df_filtered['day_of_week'] = df_filtered['post_time'].dt.day_name()
            grouped_data = df_filtered.groupby(['day_of_week', 'content_type']).agg({'engagement':'sum'}).reset_index()
            pivot_df = grouped_data.pivot_table(index='day_of_week', columns='content_type', values='engagement', aggfunc='sum', fill_value=0)

            st.subheader('Stacked Bar')

            chart = alt.Chart(grouped_data).mark_bar().encode(
                x='day_of_week',
                y='engagement',
                color='content_type',
                tooltip=['day_of_week', 'content_type', 'engagement']
            ).interactive()

            pivot_df.plot(kind='bar', stacked=True, figsize=(10, 6))
            plt.xlabel('Day of the Week')
            plt.ylabel('Engagement')
            plt.title('Engagement by Day of the Week and Content Type')

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("Altair Chart")
                st.altair_chart(chart, use_container_width=True)

            with col2:
                st.markdown("Matplotlib Chart")
                st.pyplot(plt)

            with col3:
                st.markdown("Streamlit Chart")
                st.bar_chart(pivot_df)

            df_scat = content_data_slicing_copy[(content_data_slicing_copy['likes_count'] > 0)]

            st.subheader('Scatter Plot')

            chart_1 = alt.Chart(df_scat).mark_point().encode(
                x='comments_count',
                y='likes_count',
                color='content_type',
                tooltip=['content_type', 'likes_count', 'comments_count']
            ).interactive()

            plt.figure(figsize=(10, 6))
            plt.scatter(df_filtered['comments_count'], df_filtered['likes_count'])
            plt.xlabel('comments_count')
            plt.ylabel('likes_count')
            
            chart11, chart22 = st.columns(2)

            with chart11:
                st.markdown("Altair Chart")
                st.altair_chart(chart_1, use_container_width=True)

            with chart22:
                st.markdown("Matplotlib Chart")
                st.pyplot(plt)

# Streamlit app starts here
if __name__ == '__main__':
    with st.sidebar.container():
        selected_page = option_menu(
            menu_title='ALVA X Purwadhika Workshop',
            options=['Case 1','Case 2'],
            styles={
                "nav-link-selected": {"background-color": "#32b280"}, ##7605c1
            }
        )

    if selected_page == 'Case 1':   
        predict_property_value()
    
    elif selected_page == 'Case 2':
        st.session_state.estimate_button = False
        st.session_state.property_info = False
        data_viz()


