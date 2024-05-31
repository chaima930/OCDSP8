import streamlit as st
import requests
import numpy as np 
import pandas as pd
import pyarrow.parquet as pq
import plotly.express as px
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import seaborn as sns

# Initialize session state keys
if 'btn_clicked' not in st.session_state:
    st.session_state['btn_clicked'] = False
if 'btn_clicked2' not in st.session_state:
    st.session_state['btn_clicked2'] = False

def callback1():
    st.session_state['btn_clicked'] = True
    st.session_state['btn_clicked2'] = False

def callback2():
    st.session_state['btn_clicked'] = False
    st.session_state['btn_clicked2'] = True

def credit_score_gauge(score):
    colors = ['#FF0000', '#FFFF00', '#00FF00']
    thresholds = [0, 0.5, 1]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", list(zip(thresholds, colors)))
    norm = mcolors.Normalize(vmin=0, vmax=1)
    fig, ax = plt.subplots(figsize=(6, 0.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 1, 0, 0.5])
    for i, threshold in enumerate(thresholds):
        ax.plot([threshold, threshold], [0.45, 0.5], color='black')
        ax.text(threshold, 0.55, str(threshold), fontsize=12, ha='center', va='bottom', color='black')
    ax.plot([0.5, 0.5], [0, 0.5], linestyle='--', color='black', label='Threshold')
    ax.plot([score, score], [0, 0.5], color='black', linewidth=2, label='Client score')
    ax.text(score, -0.7, f'{score:.3f}', fontsize=14, ha='center', va='bottom', color='black')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), fancybox=True, shadow=True, ncol=2)
    st.pyplot(fig, clear_figure=True)

def get_prediction(data):
    api_url = "https://card-4873eb75da10.herokuapp.com/predict"  # Update with your API URL
    df_test = {'df_test': data.drop(columns=['SK_ID_CURR']).values.tolist()}
    try:
        response = requests.post(api_url, json=df_test)
        response.raise_for_status()
        result = response.json()
        
        if 'prediction' not in result:
            st.error("The 'prediction' key is missing from the API response.")
            return None, None
        
        prediction_score = result['prediction'][0]
        prediction_result = 'Credit accepted' if prediction_score > 0.5 else 'Credit denied'
        return prediction_result, prediction_score
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None, None
    except ValueError as e:
        st.error(f"Error parsing API response: {e}")
        return None, None

def bivariate_analysis(feature1, feature2):
    fig, ax = plt.subplots()
    filtered_df = df_train[df_train['TARGET'] == int(prediction_result == 'Credit denied')]
    if filtered_df[feature1].dtype == 'int64' and filtered_df[feature2].dtype == 'int64':
        sns.countplot(data=filtered_df, x=feature1, hue=feature2, ax=ax)
    elif filtered_df[feature1].dtype != 'int64' and filtered_df[feature2].dtype != 'int64':
        sns.scatterplot(data=filtered_df, x=feature1, y=feature2, ax=ax)
    else:
        if filtered_df[feature1].dtype == 'int64':
            categorical_feature, continuous_feature = feature1, feature2
        else:
            categorical_feature, continuous_feature = feature2, feature1
        sns.boxplot(data=filtered_df, x=categorical_feature, y=continuous_feature, ax=ax)
    ax.set_title(f'Bivariate Analysis between {feature1} and {feature2}')
    st.pyplot(fig, clear_figure=True)

# Load sample parquet data (replace this with your actual data loading)
parquet_file = 'data/df_test.parquet'
table = pq.read_table(parquet_file)
df = table.to_pandas()
df = df.drop("index", axis=1)
df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)

train_data = 'data/df_train.parquet'
train_table = pq.read_table(train_data)
df_train = train_table.to_pandas()
df_train = df_train.drop("index", axis=1)
df_train['SK_ID_CURR'] = df_train['SK_ID_CURR'].astype(int)

# Ensure the test and train data have the same features
train_columns = df_train.columns.drop(['TARGET'])
df = df[train_columns]

# Streamlit app
st.title('Credit Scoring Prediction')

# Dropdown for client IDs
selected_client_id = st.selectbox('Select Client ID:', df['SK_ID_CURR'].unique())

# Display selected client's data
st.subheader('Selected Client Data:')
selected_client_data = df.loc[df['SK_ID_CURR'] == selected_client_id]
st.write(selected_client_data)

# Button to trigger prediction
if st.button('Predict', on_click=callback1) or st.session_state['btn_clicked']:
    # Make API request and get prediction
    prediction_result, prediction_score = get_prediction(selected_client_data)

    # Display prediction result
    st.subheader('Prediction Result:')
    if prediction_result is not None:
        st.write(f"The credit status is: {prediction_result}")
        st.write(f"The prediction score is: {prediction_score:.2%}")
        credit_score_gauge(prediction_score)
        if prediction_score < 0.5:
            pred_res = 0
        else:
            pred_res = 1

        # Add visualization of features
        st.subheader('Visualization of Client Features:')
        selected_feature = st.selectbox('Select Feature:', df_train.drop(['SK_ID_CURR', 'TARGET'], axis=1).columns)
        vertical_line_value = selected_client_data[selected_feature].values[0]
        fig = px.histogram(df_train.loc[df_train['TARGET'] == pred_res], x=selected_feature, title=f'Distribution of {selected_feature}')
        fig.add_vline(x=vertical_line_value, line_dash="dash", line_color="red", annotation_text=f"Vertical Line at {vertical_line_value}")
        st.plotly_chart(fig)

        # Add comparison of clients
        st.subheader('Comparison with Other Clients:')
        selected_feature_comparison = st.selectbox('Select Feature for Comparison:', df.columns)
        fig_comparison = px.histogram(df, x=selected_feature_comparison, title=f'Comparison of {selected_feature_comparison} with Other Clients')
        st.plotly_chart(fig_comparison)

        # Graphique d’analyse bi-variée entre deux features sélectionnées
        st.subheader('Bi-variate Analysis:')
        # Select two features for bivariate analysis
        selected_feature1 = st.selectbox('Select Feature 1:', df.drop(columns=['SK_ID_CURR']).columns, key='feature_selection1_mod')
        selected_feature2 = st.selectbox('Select Feature 2:', df.drop(columns=['SK_ID_CURR']).columns, key='feature_selection2_mod')

        # Display bivariate analysis
        bivariate_analysis(selected_feature1, selected_feature2)
        st.text("A graphical analysis of the relationship between two selected features for the same target as the client.")
