import os
import joblib
import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import time
from datetime import datetime
import calendar
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import uuid

# from langchain.llms import LlamaCpp
from langchain_community.llms import LlamaCpp
# from langchain.embeddings import LlamaCppEmbeddings
from langchain_community.embeddings import LlamaCppEmbeddings

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# from langchain.chains import ConversationChain
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
import sqlite3
import os
import re

# Set page configuration
st.set_page_config(
    page_title="Marketing Research Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better UI
st.markdown("""
    <style>

    .main-header {
        font-size: 26px;
        font-weight: bold;
        color: #4CAF50;
        margin-bottom: 10px;
    } 

    .main-title {
        font-size: 28px;
        font-weight: bold;
        color: #4CAF50;
    }
    .card {
        padding: 15px;
        border-radius: 10px;
        background-color: #f9f9f9;
        margin-bottom: 10px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
    }
    .success {
        color: #2E8B57;
        font-weight: bold;
    }
    .failure {
        color: #FF6347;
        font-weight: bold;
    }

    .stRadio > div {
        display: flex;
        flex-direction: row;
        align-items: center;
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        background-color: #1e3a8a;
        color: white;
        border-radius: 0.5rem;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #4f46e5;
        color: white;
        margin-left: 2rem;
    }
    .bot-message {
        background-color: #6d28d9;
        color: white;
        margin-right: 2rem;
    }
    .message-time {
        font-size: 0.8rem;
        color: #d1d5db;
        margin-top: 0.25rem;
    }
    .main-header {
        background: linear-gradient(90deg, #1976D2, #64B5F6);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
    }
    # .prediction-card {
    #     background: white;
    #     padding: 2rem;
    #     border-radius: 1rem;
    #     box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    # }
    </style>
""", unsafe_allow_html=True)

def load_models():
    try:
        with open('encoder.pkl', 'rb') as f:
            loaded_encoders = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        return loaded_encoders, loaded_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def handle_missing_values(df):
    for col in df.select_dtypes(include=['float64', 'int64']):
        df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include=['object']):
        df[col].fillna('Unknown', inplace=True)
    return df

def prepare_and_predict(input_data, encoders, model):
    try:
        X_new = pd.DataFrame([input_data])
        for col in X_new.select_dtypes(include='object'):
            if col in encoders:
                encoder = encoders[col]
                X_new[col] = X_new[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
            else:
                X_new[col] = -1
        prediction = model.predict(X_new).astype(bool)
        probabilities = model.predict_proba(X_new)[0]
        return prediction[0], probabilities
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None

def visualize_predictions(predictions):
    st.subheader("📊 Overall Predictions Insights")
    results_df = pd.DataFrame(predictions)
    result_counts = results_df['Prediction'].value_counts().rename(index={True: 'Success', False: 'Failure'})
    fig = go.Figure(go.Pie(
        labels=result_counts.index,
        values=result_counts.values,
        marker=dict(colors=['#FF4C4C', '#4CAF50']),
        hole=0.5
    ))
    fig.update_layout(title="Prediction Distribution", template="plotly_white")
    
    unique_key = str(uuid.uuid4())# Generate a unique key for the chart
    st.plotly_chart(fig, use_container_width=True, key=unique_key)

    def color_confidence(val, success=True):
        color_intensity = int(255 * val / 100)
        color = f'rgba({0 if success else color_intensity}, {color_intensity if success else 0}, 0, 0.6)'
        return f'background-color: {color}'

    st.subheader("📋 All 'Print Related' Purchase Predictions")
    styled_df = results_df.style.applymap(lambda v: color_confidence(v, success=True) if isinstance(v, float) and 'Confidence Success' in results_df.columns else '', subset=['Confidence Success'])
    styled_df = styled_df.applymap(lambda v: color_confidence(v, success=False) if isinstance(v, float) and 'Confidence Failure' in results_df.columns else '', subset=['Confidence Failure'])
    styled_df = styled_df.applymap(lambda v: 'color: #4CAF50' if v == True else 'color: #FF4C4C', subset=['Prediction'])
    st.dataframe(styled_df, use_container_width=True)

# Set your Google API key
os.environ['GOOGLE_API_KEY'] = 'add your own XD'

# Initialize Gemini model
llm = GoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0.7,
    n_ctx=4096,
    max_output_tokens=2048,
    top_p=0.8,
    verbose=True
)

db = SQLDatabase.from_uri("sqlite:///chatbot.sqlite")  # Update with your DB path

# Create a template for SQL query generation
sql_prompt = PromptTemplate(
    template="""Table info: {schema}
Question: {question}
Write a SQL query to answer this question:""",
    input_variables=["schema", "question"]
)
# Create chain for SQL query generation
sql_chain = create_sql_query_chain(llm, db)

response_prompt = PromptTemplate(
    template="""Question: {question}
Data: {result}
Give a brief answer:""",
    input_variables=["question", "result"]
)
class SQLChatbot:
    def __init__(self):
        self.memory = ConversationBufferMemory(return_messages=True)
        self.conversation = RunnableWithMessageHistory(
            runnable=llm,  # Assigning the LLM model
            get_session_history=lambda session_id: self.memory,  # Function to retrieve history
            verbose=True
        )

    
    def get_table_schema(self):
        """Get the schema of all tables in the database"""
        tables = db.get_table_info()
        return "\n".join(tables)
    
    def clean_sql_query(self, query):
        """Clean the SQL query by removing markdown code blocks and fixing table names"""
        # Remove markdown code block markers
        query = re.sub(r'```\w*\n?', '', query)
        query = query.strip()
        # Replace incorrect table name if present
        query = query.replace('chatbot_data', 'chatbot')
        return query
    
    def execute_sql_query(self, query):
        """Execute SQL query and return results"""
        try:
            return db.run(query)
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    def process_question(self, user_question):
        """Process user question and return response"""
        try:
            # Get table schema
            schema = self.get_table_schema()
            
            # Generate SQL query
            sql_query = sql_chain.invoke({
                "schema": schema,
                "question": user_question
            })
            
             # Clean the SQL query
            cleaned_query = self.clean_sql_query(sql_query)
            
            # Execute query
            query_result = self.execute_sql_query(cleaned_query)
            
            # Generate natural language response
            response = llm.invoke(response_prompt.format(
                question=user_question,
                result=str(query_result)
            ))

            return {
                "response": response,
                "sql_query": cleaned_query,
                "query_result": query_result
            }
        
        except Exception as e:
            return {
                "error": f"An error occurred: {str(e)}"
            }

def chat_interface():
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    st.sidebar.markdown("### ✨ AI Assistant")
    chatbot = SQLChatbot()
    # Chat input
    with st.sidebar:
        query = st.text_input("Ask me anything about the predictions...", key="chat_input", placeholder="Fire your business related query here!")
        
        if st.button("Send", key="send_button"):
            if query:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": query, "time": datetime.now().strftime("%H:%M")})
                result = chatbot.process_question(query)
                response=result["response"]
                # response=result["response"]+ result["sql_query"] + result["query_result"]
                # Generate response based on query
                # if "accuracy" in query.lower():
                #     response = "The model accuracy is 85% based on our test data. This means we can reliably predict sales outcomes in most cases."
                # elif "confidence" in query.lower():
                #     response = "Our model provides confidence scores for both success and failure predictions, helping you make informed decisions."
                # elif "help" in query.lower():
                #     response = "I can help you understand the predictions, model accuracy, and confidence scores. Feel free to ask specific questions!"
                # else:
                #     response = "I'm here to help! You can ask about model accuracy, confidence scores, or specific predictions."
                
                # Add bot response with slight delay for natural feel
                time.sleep(0.5)
                st.session_state.messages.append({"role": "assistant", "content": response, "time": datetime.now().strftime("%H:%M")})

        # Display chat history
        st.markdown("#### Chat History")
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                    <div class='chat-message user-message'>
                        <div>👤 You: {message["content"]}</div>
                        <div class='message-time'>{message["time"]}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='chat-message bot-message'>
                        <div>🤖 Assistant: {message["content"]}</div>
                        <div class='message-time'>{message["time"]}</div>
                    </div>
                """, unsafe_allow_html=True)

def load_and_prepare_data():
    try:
        df = pd.read_excel('./data/sales_data.xlsx')
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Month'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_sales_profit_analysis(df):
    fig = make_subplots(
    rows=3, cols=2,  # Accommodating additional graphs
    subplot_titles=(
        "Monthly Sales Trend", "Product Category Sales",
        "Profit Margins by Category", "Sales vs Profit Correlation",
        "Yearly Sales Overview", "Profit by Region"
    ),
    vertical_spacing=0.15,  # Increase vertical space
    horizontal_spacing=0.2   # Increase horizontal space
)

    # Monthly Sales Trend
    monthly_sales = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
    monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(DAY=1))
    fig.add_trace(
        go.Scatter(
            x=monthly_sales['Date'], y=monthly_sales['Sales'],
            mode='lines+markers', name='Monthly Sales',
            line=dict(color='#4CAF50')
        ),
        row=1, col=1
    )

    # Product Category Sales
    category_sales = df.groupby('Category')['Sales'].sum().reset_index()
    fig.add_trace(
        go.Bar(
            x=category_sales['Category'], y=category_sales['Sales'],
            name='Category Sales', marker_color='#2196F3'
        ),
        row=1, col=2
    )

    # Profit Margins by Category
    fig.add_trace(
        go.Bar(
            x=df['Category'].unique(),
            y=df.groupby('Category')['Profit'].mean(),
            name='Avg Profit Margin', marker_color='#FF9800'
        ),
        row=2, col=1
    )

    # Sales vs Profit Correlation
    fig.add_trace(
        go.Scatter(
            x=df['Sales'], y=df['Profit'],
            mode='markers', name='Sales vs Profit',
            marker=dict(color='#E91E63', size=8)
        ),
        row=2, col=2
    )

    # Yearly Sales Overview
    yearly_sales = df.groupby('Year')['Sales'].sum().reset_index()
    fig.add_trace(
        go.Bar(
            x=yearly_sales['Year'], y=yearly_sales['Sales'],
            name='Yearly Sales', marker_color='#673AB7'
        ),
        row=3, col=1
    )

    # Profit by Region
    if 'Region' in df.columns:
        region_profit = df.groupby('Region')['Profit'].sum().reset_index()
        fig.add_trace(
            go.Bar(
                x=region_profit['Region'], y=region_profit['Profit'],
                name='Profit by Region', marker_color='#009688'
            ),
            row=3, col=2
        )

    # Shift the right-side graphs to the right using the domain property
    fig.update_xaxes(domain=[0.55, 1.0], row=1, col=2)
    fig.update_xaxes(domain=[0.55, 1.0], row=2, col=2)
    fig.update_xaxes(domain=[0.55, 1.0], row=3, col=2)

    # Add x and y axis labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Sales", row=1, col=1)

    fig.update_xaxes(title_text="Category", row=1, col=2)
    fig.update_yaxes(title_text="Sales", row=1, col=2)

    fig.update_xaxes(title_text="Category", row=2, col=1)
    fig.update_yaxes(title_text="Average Profit", row=2, col=1)

    fig.update_xaxes(title_text="Sales", row=2, col=2)
    fig.update_yaxes(title_text="Profit", row=2, col=2)

    fig.update_xaxes(title_text="Year", row=3, col=1)
    fig.update_yaxes(title_text="Total Sales", row=3, col=1)

    if 'Region' in df.columns:
        fig.update_xaxes(title_text="Region", row=3, col=2)
        fig.update_yaxes(title_text="Total Profit", row=3, col=2)

    # Layout settings
    fig.update_layout(
        height=1400,  # Increased to handle the added space
        showlegend=True, template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def business_insights_tab():
    st.markdown("""
        <style>
        .insight-section {
            background: #1E1E1E;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            border: 1px solid #333;
        }
        .section-header {
            color: #fff;
            font-size: 20px;
            margin-bottom: 15px;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    df = load_and_prepare_data()
    if df is None:
        return
    # st.markdown("<h3 class='section-header'>📊 Sales & Profit Analysis</h3>", unsafe_allow_html=True)
    fig_sales = create_sales_profit_analysis(df)
    st.plotly_chart(fig_sales, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    # Main header with gradient background
    st.markdown("""
        <div class='main-header'>
            <h1>📊Marketing Research Dashboard</h1>
            <p>Make data-driven decisions with our advanced ML model</p>
        </div>
    """, unsafe_allow_html=True)

    encoders, model = load_models()
    if encoders is None or model is None:
        return

    # Initialize chat interface in sidebar
    chat_interface()

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Single Prediction",
        "📁 Batch Predictions",
        "📈 Business Insights",
        "🔄 Compare Algorithms"
    ])

    with tab1:
        #st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        st.header("📝 Enter Sales Data")
        col1, col2 = st.columns(2)
        
        with col1:
            customer_id = st.text_input("Customer ID", value="SZ-20035")
            sales = st.number_input("Sales", value=0.0, format="%.2f")
            discount = st.slider("Discount", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

        with col2:
            category = st.selectbox("Category", encoders['Category'].classes_ if 'Category' in encoders else ["Technology"])
            sub_category = st.selectbox("Sub-Category", encoders['Sub-Category'].classes_ if 'Sub-Category' in encoders else ["Machines"])
            total_revenue = st.number_input("Total Revenue", value=1201.06, format="%.2f")

        input_data = {
            'Customer ID': customer_id,
            'Sales': sales,
            'Discount': discount,
            'Total Revenue': total_revenue,
            'Category': category,
            'Sub-Category': sub_category
        }

        if st.button("🔮 Predict Sales"):
                st.session_state.input_data = input_data
                st.session_state.show_prediction = True

        if 'show_prediction' in st.session_state:
            prediction, probabilities = prepare_and_predict(
                st.session_state.input_data, encoders, model
            )

            if prediction is not None:
                st.success(f"### {'✅' if prediction else '❌'} Prediction: {'Likely to purchase printer related product.' if prediction else 'Unlikely to purchase printer related product.'}")

                st.subheader("📊 Prediction Confidence Levels")
                fig = go.Figure(go.Bar(
                    x=['Failure', 'Success'],
                    y=probabilities * 100,
                    marker_color=['#FF4C4C', '#4CAF50'],
                    text=[f'{prob:.1f}%' for prob in probabilities * 100],
                    textposition='auto'
                ))
                fig.update_layout(
                    yaxis_title="Confidence (%)",
                    title="Model Confidence",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("📋 Input Summary")
                styled_table = pd.DataFrame([st.session_state.input_data]).style.set_properties(
                    **{
                        'background-color': '#1E1E1E',
                        'color': 'white',
                        'border-color': 'white',
                        'padding': '10px'
                    }
                ).set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#333333'), ('color', 'white'), ('padding', '10px')]},
                    {'selector': 'td', 'props': [('background-color', '#2A2A2A'), ('color', 'white'), ('padding', '8px')]},
                    {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#262626')]}
                ])
                st.dataframe(styled_table, use_container_width=True)

    with tab2:
        #st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        st.header("📁 Upload CSV File for Batch Predictions")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            with st.spinner("Processing batch predictions..."):
                df = pd.read_csv(uploaded_file)
                df = handle_missing_values(df)
                predictions = []
                progress_bar = st.progress(0)
                for idx, row in df.iterrows():
                    input_data = row.to_dict()
                    prediction, probabilities = prepare_and_predict(input_data, encoders, model)
                    if prediction is not None:
                        predictions.append({
                            'Customer ID': input_data.get('Customer ID', 'N/A'),
                            'Prediction': prediction,
                            'Confidence Success': probabilities[1] * 100,
                            'Confidence Failure': probabilities[0] * 100
                        })
                    progress_bar.progress((idx + 1) / len(df))
                visualize_predictions(predictions)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.title("📈Business Predictions")
        # Nested Tabs for Business Insights
        insights_tabs = st.tabs([
            "Sales and Profit Analysis",
            "Time-Based Insights",
            "Profitability Metrics",
            "Customer Behavior Analysis"
        ])

        df = pd.read_excel("./data/sales_data.xlsx")
        # Sales and Profit Analysis
        with insights_tabs[0]:
            st.subheader("Sales and Profit Analysis")
            business_insights_tab()
            st.markdown("</div>", unsafe_allow_html=True)

        # Time-Based Insights
        with insights_tabs[1]:
            st.subheader("Time Based Insights")     
            monthly_sales = df.groupby('Month').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
            fig = px.line(monthly_sales, x='Month', y=['Sales', 'Profit'], title="Monthly Sales vs Profit")
            st.plotly_chart(fig)

            yearly_sales = df.groupby('Year').agg({'Sales': 'sum'}).reset_index()
            fig2 = px.bar(yearly_sales, x='Year', y='Sales', title="Yearly Sales")
            st.plotly_chart(fig2)
        

        # Profitability Metrics
        with insights_tabs[2]:
            st.subheader("Profitability Metrics")
            profit_margin = df.groupby('Category').agg({'Profit Margin': 'mean'}).reset_index()
            fig3 = px.bar(profit_margin, x='Category', y='Profit Margin', title="Profit Margin by Category", color='Profit Margin')
            st.plotly_chart(fig3)

            discount_sales = df.groupby('Sub-Category').agg({'Discounted Sales': 'sum'}).reset_index()
            fig4 = px.pie(discount_sales, names='Sub-Category', values='Discounted Sales', title="Discounted Sales Distribution")
            st.plotly_chart(fig4)

        # Customer Behavior Analysis
        with insights_tabs[3]:
            st.subheader("Customer Behaviour Insights")
            # Check for boolean values
            if df['High Value Customer'].dtype == bool:
                high_value = df[df['High Value Customer'] == True]
            else:
                high_value = df[df['High Value Customer'].astype(str).str.strip().str.upper() == 'TRUE']

            # Group by Region
            high_value_grouped = high_value.groupby('Region').size().reset_index(name='Count')

            # Plot if there are values
            if not high_value_grouped.empty:
                fig5 = px.bar(high_value_grouped, x='Region', y='Count', title="High-Value Customers by Region", color='Count')
                st.plotly_chart(fig5)
            else:
                st.write("No high-value customers found in any region.")
            # high_value = df[df['High Value Customer'] == 'TRUE'].groupby('Region').size().reset_index(name='Count')
            # fig5 = px.bar(high_value, x='Region', y='Count', title="High-Value Customers by Region", color='Count')
            # st.plotly_chart(fig5)

            segment_sales = df.groupby('Segment').agg({'Sales': 'sum'}).reset_index()
            fig6 = px.pie(segment_sales, names='Segment', values='Sales', title="Sales by Customer Segment")
            st.plotly_chart(fig6)
                

       
        with tab4:
            # 📊 Model Metrics Data
            metrics_data = [
                {"Model": "Decision Tree", "Accuracy": 0.9102, "Recall": 1.0, "Precision": 0.08, "F1 Score": 0.15},
                {"Model": "XGBoost", "Accuracy": 0.9481, "Recall": 0.8636, "Precision": 0.12, "F1 Score": 0.20},
                {"Model": "Random Forest", "Accuracy": 0.9151, "Recall": 0.9545, "Precision": 0.08, "F1 Score": 0.15},
                {"Model": "Gradient Boosting", "Accuracy": 0.9102, "Recall": 1.0, "Precision": 0.08, "F1 Score": 0.15}
            ]

            # Convert to DataFrame
            metrics_df = pd.DataFrame(metrics_data)

            # 📊 Visualize Model Metrics
            st.markdown("<div class='main-header'>🤖 Algorithm Comparison</div>", unsafe_allow_html=True)
            
            fig = go.Figure()

            # Add bars for each metric
            for metric in ["Accuracy", "Recall", "Precision", "F1 Score"]:
                fig.add_trace(go.Bar(
                    x=metrics_df["Model"],
                    y=metrics_df[metric],
                    name=metric
                ))

            # Update chart layout
            fig.update_layout(
                title="🔍 Model Comparison Based on Metrics",
                xaxis_title="Models",
                yaxis_title="Scores",
                barmode='group',
                template='plotly_dark',
                height=600
            )

            # Highlight XGBoost as the best model
            best_model = metrics_df.loc[metrics_df['Accuracy'].idxmax(), 'Model']
            st.success(f"🏆 **{best_model}** outperforms other models with the highest accuracy.")

            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()