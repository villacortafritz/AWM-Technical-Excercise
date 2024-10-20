import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import io
import warnings
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# Step 1: Data Manipulation and Cleaning

# Loading Raw CSV from a URL with error handling
def load_data(url):
    response = requests.get(url, verify=False)
    df = pd.read_csv(io.StringIO(response.text), on_bad_lines='skip', engine='python')
    print("First few rows of the DataFrame:\n", df.head())  # Display the first few rows
    print("Columns in the DataFrame:", df.columns)  # Display column names
    return df

# Clean Data with handling for column discrepancies
def clean_data(df):
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    if 'Sales Value' in df.columns:
        df.loc[:, 'Sales Value'] = df['Sales Value'].fillna(df['Sales Value'].mean())
    else:
        raise ValueError("Column 'Sales Value' not found in the DataFrame.")
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

#Calculate average sales per region
def calculate_average_sales_per_region(df):
    avg_sales_per_region = df.groupby('Region')['Sales Value'].mean().reset_index()
    avg_sales_per_region.rename(columns={'Sales Value': 'Average Sales Value'}, inplace=True)
    return avg_sales_per_region

#Filter Regions per Set Threshold (100,000)
def filter_regions_by_sales_threshold(df, threshold=100000):
    total_sales_per_region = df.groupby('Region')['Sales Value'].sum().reset_index()
    return df, total_sales_per_region

#Identify Top 3 Regions by Total Sales Volume
def top_3_products_by_region(df):
    top_products = df.groupby(['Region', 'Product'], as_index=False)['Sales Volume'].sum()
    top_3_products = pd.DataFrame()
    for region in top_products['Region'].unique():
        region_top_products = top_products[top_products['Region'] == region].nlargest(3, 'Sales Volume')
        top_3_products = pd.concat([top_3_products, region_top_products])
    top_3_products.reset_index(drop=True, inplace=True)
    return top_3_products

# Step 2: Generate Detailed Business Insights

def generate_detailed_insights(df):
    total_sales_by_region = df.groupby('Region')['Sales Value'].sum().reset_index()
    top_region = total_sales_by_region.sort_values(by='Sales Value', ascending=False).iloc[0]
    
    total_sales_by_product = df.groupby('Product')['Sales Value'].sum().reset_index()
    top_product = total_sales_by_product.sort_values(by='Sales Value', ascending=False).iloc[0]

    df['Month'] = pd.to_datetime(df['Date'], errors='coerce').dt.to_period('M')
    monthly_sales_trend = df.groupby('Month')['Sales Value'].sum().reset_index()
    peak_month = monthly_sales_trend.sort_values(by='Sales Value', ascending=False).iloc[0]

    insights = (
        "<h2>Business Insights</h2>"
        "<ul>"
        f"<li><strong>Top Region:</strong> {top_region['Region']} has the highest sales, totaling <strong>{top_region['Sales Value']:,}</strong>. "
        "Focus marketing efforts and inventory optimization in this region to sustain high performance.</li>"
        
        f"<li><strong>Top Product:</strong> {top_product['Product']} is the best-selling product with total sales of <strong>{top_product['Sales Value']:,}</strong>. "
        "Consider increasing production and promotions for this product to maintain demand.</li>"
        
        f"<li><strong>Peak Sales Month:</strong> Sales peaked in <strong>{peak_month['Month']}</strong>, reaching <strong>{peak_month['Sales Value']:,}</strong>. "
        "Align marketing campaigns with this period to maximize impact.</li>"
        
        "<li><strong>Underperforming Regions:</strong> Regions with lower total sales require targeted marketing or promotions to boost demand.</li>"
        
        "<li><strong>Sales Consistency:</strong> Sales performance varies across months, with clear peaks. Implement strategies to maintain momentum year-round.</li>"
        
        "<li><strong>Customer Demographics:</strong> The dataset lacks customer demographic data, limiting a full analysis of customer behavior. "
        "Future data collection should include demographics to enhance customer insights.</li>"
        "</ul>"
    )
    return insights

# Step 3: Generate Bar Graph Including All Regions

def generate_bar_graph(total_sales_per_region, threshold):
    # Set a static color for each region
    colors = ['red' if region == 'South' else 'blue' for region in total_sales_per_region['Region']]

    fig = go.Figure(data=[
        go.Bar(
            x=total_sales_per_region['Region'],
            y=total_sales_per_region['Sales Value'],
            marker_color=colors,
            text=total_sales_per_region['Sales Value'].apply(lambda x: f"{x:,.0f}"),
            textposition='outside'
        )
    ])

    # Update layout and add title
    fig.update_layout(
        title='Total Sales by Region (with Threshold)',
        xaxis_title='Region',
        yaxis_title='Total Sales Value',
        template='plotly_white'
    )

    # Add a dashed line to indicate the threshold
    fig.add_shape(
        type='line',
        x0=-0.5, x1=len(total_sales_per_region) - 0.5,
        y0=threshold, y1=threshold,
        line=dict(color='Black', width=2, dash='dash'),
        xref='x', yref='y'
    )
    fig.add_annotation(
        x=len(total_sales_per_region) - 1,
        y=threshold,
        text=f"Threshold: {threshold}",
        showarrow=False,
        yshift=10,
        font=dict(color='Black')
    )

    return fig

# Step 4: Generate Line Graph for Sales Trends with Data Labels

def generate_line_graph(df):
    df['Month'] = pd.to_datetime(df['Date'], errors='coerce').dt.to_period('M').astype(str)
    monthly_sales = df.groupby('Month')['Sales Value'].sum().reset_index()

    fig = px.line(
        monthly_sales,
        x='Month',
        y='Sales Value',
        title='Total Sales Trends Over Time',
        labels={'Sales Value': 'Total Sales Value'},
        template='plotly_white'
    )

    # Add data labels within the graph with comma separator
    fig.update_traces(text=monthly_sales['Sales Value'].apply(lambda x: f"{x:,.0f}"), textposition='top center', mode='lines+markers+text')

    return fig

# Step 5: Create an HTML Report

def create_html_report(df, total_sales_per_region, filepath, threshold):
    avg_sales_per_region = calculate_average_sales_per_region(df)
    top_3_products = top_3_products_by_region(df)

    bar_graph = generate_bar_graph(total_sales_per_region, threshold)
    line_graph = generate_line_graph(df)

    avg_sales_table = go.Figure(data=[go.Table(
        header=dict(values=["Region", "Average Sales Value"],
                    fill_color='paleturquoise',
                    align='left',
                    height=30),
        cells=dict(values=[avg_sales_per_region['Region'], avg_sales_per_region['Average Sales Value']],
                   fill_color='lavender',
                   align='left',
                   height=20))
    ])

    # Calculate dynamic height for the top 3 products table
    top_3_table_height = 30 + len(top_3_products) * 20

    top_3_table = go.Figure(data=[go.Table(
        header=dict(values=["Region", "Product", "Sales Volume"],
                    fill_color='paleturquoise',
                    align='left',
                    height=30),
        cells=dict(values=[top_3_products['Region'], top_3_products['Product'], top_3_products['Sales Volume']],
                   fill_color='lavender',
                   align='left',
                   height=20))
    ])

    avg_sales_table.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=200)
    top_3_table.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=top_3_table_height)

    insights = generate_detailed_insights(df)
    
    with open(filepath, "w") as f:
        f.write("<html><head><title>Affordable Wire Management (AWM)</title>")
        f.write("<style>body { font-family: Arial, sans-serif; text-align: center; }")
        f.write("h1.biggest { font-size: 36px; font-weight: bold; margin-bottom: 5px; }")
        f.write("h1.big { font-size: 28px; font-weight: bold; margin-bottom: 5px; }")
        f.write("h1.small { font-size: 18px; font-weight: normal; margin-top: 0; }")
        f.write("h2 { margin-top: 20px; font-size: 24px; }</style></head><body>")
        
        # Add the custom header with a hierarchy
        f.write("<h1 class='biggest'>Affordable Wire Management (AWM)</h1>")
        f.write("<h1 class='big'>Optional Python Assessment - Web Version</h1>")
        f.write("<h1 class='small'>Prepared by Fritz Laurence Villacorta</h1>")

        # Add business insights
        f.write(insights)

        # Add the title for the bar chart
        f.write("<h2>Total Sales by Region (with Threshold)</h2>")
        f.write(bar_graph.to_html(full_html=False, include_plotlyjs='cdn'))

        # Add the title for the line chart
        f.write("<h2>Total Sales Trends Over Time</h2>")
        f.write(line_graph.to_html(full_html=False, include_plotlyjs='cdn'))
        
        # Add the title for the average sales table
        f.write("<h2>Average Sales per Region</h2>")
        f.write(avg_sales_table.to_html(full_html=False, include_plotlyjs='cdn'))

        # Add the title for the top 3 products table
        f.write("<h2>Top 3 Products by Region</h2>")
        f.write(top_3_table.to_html(full_html=False, include_plotlyjs='cdn'))

        f.write("</body></html>")
    
    print(f"Report generated and saved to {filepath}")

# Main execution

def run_all(filepath, report_filepath, threshold=100000):
    df = load_data(filepath)
    df = clean_data(df)
    df, total_sales_per_region = filter_regions_by_sales_threshold(df, threshold)
    create_html_report(df, total_sales_per_region, report_filepath, threshold)

if __name__ == '__main__':
    data_url = 'https://raw.githubusercontent.com/villacortafritz/AWM-Technical-Excercise/refs/heads/main/AWM%20Data%20-%20Fritz%20Villacorta%20-%20Raw%20Python%20Data.csv'
    report_filepath = 'index.html'
    run_all(data_url, report_filepath, threshold=100000)