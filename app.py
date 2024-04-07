import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objs as go



def main():
    df = pd.read_csv('https://raw.githubusercontent.com/amirhossein-dezhboro/DMWithRiskAnalysisProject/main/finaldf.csv')
    df = df.rename(columns={"acceleration..0.100.": "Acceleration"})
    df = df.rename(columns={"Price.DE.": "Price"})
    df = df.rename(columns={"Efficiency_Wh_km": "Efficiency"})

    st.set_page_config(page_title='Whats my EV?', layout='wide')
    tab1, tab2, tab3, tab4 = st.tabs(["Preference Model", "Weights", "Results", "About"])

    with tab1:
        st.write('Please select the drive type of your vehicle:')
        a1 = st.checkbox('All Wheel Drive')
        a2 = st.checkbox('Front Wheel Drive')
        a3 = st.checkbox('Rear Wheel Drive')

        if st.button('Analyze'):
            dflist = []
            d1 = df[df['Drive']=='All Wheel Drive']
            d2 = df[df['Drive']=='Front Wheel Drive']
            d3 = df[df['Drive']=='Rear Wheel Drive']
            if a1 == True:
                dflist.append(d1)
            if a2 == True:
                dflist.append(d2)
            if a3 == True:
                dflist.append(d3)
            df = pd.concat(dflist, ignore_index=True)
        else:
            st.write("Please select your preference and hit Analyze")
            

# -----------------------
                        
        min_Battery, max_Battery = df['Battery'].min(), df['Battery'].max()
        Battery_range = st.slider(
            "Select Battery Range",
            min_value=min_Battery,
            max_value=max_Battery,
            value=(min_Battery, max_Battery))
        df['Battery' + '_utility'] = (df['Battery'] - df['Battery'].min()) / (df['Battery'].max() - df['Battery'].min())
        filtered_df = df[(df['Battery'] >= Battery_range[0]) & (df['Battery'] <= Battery_range[1])]
        start_point = {'x': Battery_range[0], 'y': 0}  # Starting point coordinates
        end_point = {'x': Battery_range[1], 'y': 1}    # Ending point coordinates
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=[start_point['x'], end_point['x']],
                                y=[start_point['y'], end_point['y']],
                                mode='markers+lines',  # This will create both markers and lines
                                marker=dict(size=10, color='red'),  # Customizes the marker appearance
                                line=dict(dash='dash'),  # Makes the line dashed
                                name='Line'))
        fig1.update_layout(title='Battery Capacity Utility',
                        xaxis_title='Battery Cap',
                        yaxis_title='Utility',
                        showlegend=True)
        # Display the figure in Streamlit
        st.plotly_chart(fig1)

        min_Range, max_Range= df['Range'].min(), df['Range'].max()
        Range_range = st.slider(
            "Select Range Range!",
            min_value=min_Range,
            max_value=max_Range,
            value=(min_Range, max_Range))
        df['Range' + '_utility'] = (df['Range'] - df['Range'].min()) / (df['Range'].max() - df['Range'].min())
        filtered_df = df[(df['Range'] >= Range_range[0]) & (df['Range'] <= Range_range[1])]
        start_point = {'x': Range_range[0], 'y': 0}  # Starting point coordinates
        end_point = {'x': Range_range[1], 'y': 1}    # Ending point coordinates
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=[start_point['x'], end_point['x']],
                                y=[start_point['y'], end_point['y']],
                                mode='markers+lines',  # This will create both markers and lines
                                marker=dict(size=10, color='red'),  # Customizes the marker appearance
                                line=dict(dash='dash'),  # Makes the line dashed
                                name='Line'))
        fig2.update_layout(title='Maximum Range Utility',
                        xaxis_title='Range',
                        yaxis_title='Utility',
                        showlegend=True)
        # Display the figure in Streamlit
        st.plotly_chart(fig2)

        min_Acceleration, max_Acceleration = df['Acceleration'].min(), df['Acceleration'].max()
        Acceleration_range = st.slider(
            "Select Acceleration Range",
            min_value=min_Acceleration,
            max_value=max_Acceleration,
            value=(min_Acceleration, max_Acceleration))
        df['Acceleration' + '_utility'] = (df['Acceleration'] - df['Acceleration'].min()) / (df['Acceleration'].max() - df['Acceleration'].min())
        filtered_df = df[(df['Acceleration'] >= Acceleration_range[0]) & (df['Acceleration'] <= Acceleration_range[1])]
        start_point = {'x': Acceleration_range[0], 'y': 0}  # Starting point coordinates
        end_point = {'x': Acceleration_range[1], 'y': 1}    # Ending point coordinates
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=[start_point['x'], end_point['x']],
                                y=[start_point['y'], end_point['y']],
                                mode='markers+lines',  # This will create both markers and lines
                                marker=dict(size=10, color='red'),  # Customizes the marker appearance
                                line=dict(dash='dash'),  # Makes the line dashed
                                name='Line'))
        fig3.update_layout(title='Acceleration Utility',
                        xaxis_title='Acceleration',
                        yaxis_title='Utility',
                        showlegend=True)
        # Display the figure in Streamlit
        st.plotly_chart(fig3)

        min_Efficiency, max_Efficiency = df['Efficiency'].min(), df['Efficiency'].max()
        Efficiency_range = st.slider(
            "Select Efficiency Range",
            min_value=min_Efficiency,
            max_value=max_Efficiency,
            value=(min_Efficiency, max_Efficiency))
        df['Efficiency' + '_utility'] = 1 - ((df['Efficiency'] - df['Efficiency'].min()) / (df['Efficiency'].max() - df['Efficiency'].min()))
        filtered_df = df[(df['Efficiency'] >= Efficiency_range[0]) & (df['Efficiency'] <= Efficiency_range[1])]
        start_point = {'x': Efficiency_range[0], 'y': 1}  # Starting point coordinates
        end_point = {'x': Efficiency_range[1], 'y': 0}    # Ending point coordinates
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=[start_point['x'], end_point['x']],
                                y=[start_point['y'], end_point['y']],
                                mode='markers+lines',  # This will create both markers and lines
                                marker=dict(size=10, color='red'),  # Customizes the marker appearance
                                line=dict(dash='dash'),  # Makes the line dashed
                                name='Line'))
        fig4.update_layout(title='Efficiency Utility',
                        xaxis_title='Efficiency',
                        yaxis_title='Utility',
                        showlegend=True)
        # Display the figure in Streamlit
        st.plotly_chart(fig4)

        min_Price, max_Price = df['Price'].min(), df['Price'].max()
        Price_range = st.slider(
            "Select Price Range",
            min_value=min_Price,
            max_value=max_Price,
            value=(min_Price, max_Price))
        df['Price' + '_utility'] = 1 - ((df['Price'] - df['Price'].min()) / (df['Price'].max() - df['Price'].min()))
        filtered_df = df[(df['Price'] >= Price_range[0]) & (df['Price'] <= Price_range[1])]
        start_point = {'x': Price_range[0], 'y': 1}  # Starting point coordinates
        end_point = {'x': Price_range[1], 'y': 0}    # Ending point coordinates
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=[start_point['x'], end_point['x']],
                                y=[start_point['y'], end_point['y']],
                                mode='markers+lines',  # This will create both markers and lines
                                marker=dict(size=10, color='red'),  # Customizes the marker appearance
                                line=dict(dash='dash'),  # Makes the line dashed
                                name='Line'))
        fig5.update_layout(title='Price Utility',
                        xaxis_title='Price',
                        yaxis_title='Utility',
                        showlegend=True)
        # Display the figure in Streamlit
        st.plotly_chart(fig5)
           


        
    with tab2:
        
        # Creating 5 sliders
        value1 = st.slider('Battery', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        value2 = st.slider('Range', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        value3 = st.slider('Acceleration', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        value4 = st.slider('Efficiency', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        value5 = st.slider('Price', min_value=0.0, max_value=1.0, value=0.2, step=0.01)

        # Submit button
        submit = st.button('Submit')

        # Check if submit button is pressed
        if submit:
            # Calculate the sum of all slider values
            total_sum = value1 + value2 + value3 + value4 + value5
            weights = {
            'Battery_utility': value1,
            'Acceleration_utility': value2,
            'Efficiency_utility': value3,
            'Range_utility': value4,
            'Price_utility': value5
            }
            # Check if the sum is equal to 1
            if total_sum == 1:
                st.success('The sum is equal to 1. You can proceed to the next step.')
            else:
                st.error('The sum of all values should be 1. Please adjust the sliders.')

    with tab3:
        df['Total_Utility'] = sum([df[attr] * weight for attr, weight in weights.items()])

        # Step 6: Rank vehicles based on total utility score
        df_sorted = df.sort_values(by='Total_Utility', ascending=False)
        plotdf = df_sorted[:10][["Name", "Battery_utility", 'Acceleration_utility', 'Efficiency_utility', 'Range_utility', 'Price_utility']]
        # Display the top vehicles based on utility score
        col1, col2 = st.columns(2)
        with col1:
            st.write('The top vehicles based on utility score')
            st.write(df_sorted[['Name', 'Total_Utility']].head(10))

        # Use the second column to display the Plotly chart
        with col2:
            for column, weight in weights.items():
                plotdf[column] = plotdf[column].mul(weight)
            import plotly.express as px
            import numpy as np
            # Assuming plotdf is your DataFrame
            fig = px.bar(plotdf, x='Name', y=["Battery_utility", "Acceleration_utility", "Efficiency_utility", "Range_utility", "Price_utility"],
                        title="Utilities by Name", labels={"value": "Utility Value", "variable": "Utilities"},
                        color_discrete_sequence=px.colors.qualitative.Pastel1,  # Use a nice color scheme
                        barmode='stack')  # Ensure bars are stacked

            fig.update_layout(xaxis={'categoryorder':'total descending'},  # Optional: sort bars
                            xaxis_title="Name",
                            yaxis_title="Utility Value",
                            legend_title="Utilities",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

            st.plotly_chart(fig)
            
            plotdf.set_index('Name', inplace=True)
            columns = ["Battery_utility", "Acceleration_utility", "Efficiency_utility", "Range_utility", "Price_utility"]
            num_simulations = 100
            for column in columns:
                uncertain_values = []  # to store simulation results for each record
                for index, row in plotdf.iterrows():
                    mean = row[column]
                    std_dev = mean * np.random.uniform(0.05, 0.1)  # 5-10% of the mean as std deviation                    
                    simulations = np.random.normal(mean, std_dev, num_simulations)
                    uncertain_values.append(simulations)
                plotdf["Uncertain_" + column] = uncertain_values
            
            import plotly.express as px
            import numpy as np
            import pandas as pd

            uncertain_columns = [col for col in plotdf.columns if 'Uncertain_' in col]
            plotdf['total_uncertain_utility'] = plotdf.apply(lambda row: np.sum([row[col] for col in uncertain_columns], axis=0), axis=1)

            data = []
            for index, row in plotdf.iterrows():
                for value in row['total_uncertain_utility']:
                    data.append({'Name': index, 'Total Uncertain Utility': value})
            df_long_format = pd.DataFrame(data)
            # Using Plotly Express to create the box plot
            figbox = px.box(df_long_format, x='Name', y='Total Uncertain Utility', title='Box Plot of Total Uncertain Utilities')
            # Show the plot
            st.plotly_chart(figbox)

    with tab4:
        st.write('Team members')

        st.write('Rosario Conteras')
        st.write('Saeed Ataei')
        st.write('Amirhossein Dezhboro')


if __name__ == "__main__":
    main()
