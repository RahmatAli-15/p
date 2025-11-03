# ===============================================================
# 📊 PAYTM SMART DASHBOARD — FLASK WEB APP
# ===============================================================

from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import squarify
import io, base64, os
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import random


# ===============================================================
# ⚙️ Flask App Setup
# ===============================================================
app = Flask(__name__)
DATA_PATH = os.path.join('data', 'Paytm.csv')


# ===============================================================
# 🧩 Helper Function — Convert Matplotlib Figure to Base64
# ===============================================================
def fig_to_base64(fig):
    """Converts a Matplotlib figure to a Base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return img_base64


# ===============================================================
# 🏠 ROUTE: Home Page — Date Range Selection
# ===============================================================
@app.route('/', methods=['GET', 'POST'])
def index():
    """Renders the home page with date range inputs."""
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    return render_template('index.html', min_date=min_date, max_date=max_date)


# ===============================================================
# 📈 ROUTE: Dashboard — Data Analysis and Visualization
# ===============================================================
@app.route('/dashboard', methods=['POST'])
def dashboard():
    """Displays the main analytics dashboard with insights and charts."""

    # ---------------------------------------------------------------
    # 🔹 Step 1: Load & Filter Data
    # ---------------------------------------------------------------
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')

    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Month'] = df['Date'].dt.strftime('%b-%Y')
    df['Weekday'] = df['Date'].dt.day_name()
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Month_Num'] = df['Date'].dt.month

    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    paid = filtered_df[filtered_df['TransactionType'].str.lower() == 'paid']


    # ---------------------------------------------------------------
    # 🔹 Step 2: Plotly Interactive Charts
    # ---------------------------------------------------------------
    fig1 = px.bar(
        paid.groupby('Category', as_index=False)['Amount'].sum(),
        x='Category', y='Amount', color='Category',
        title='Total Spending by Category'
    )
    category_chart = fig1.to_html(full_html=False)

    fig2 = px.pie(
        filtered_df, names='TransactionType', values='Amount',
        title='Transaction Type Breakdown'
    )
    type_chart = fig2.to_html(full_html=False)


    # ---------------------------------------------------------------
    # 🔹 Step 3: Seaborn & Matplotlib Static Charts
    # ---------------------------------------------------------------
    img_charts = {}
    sns.set_style("whitegrid")

    # 📊 Transaction count by category
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("viridis", len(paid['Category'].unique()))
    sns.countplot(
        data=paid, x='Category',
        order=paid['Category'].value_counts().index,
        palette=palette, ax=ax
    )
    ax.set_title("📊 Transaction Count by Category", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha='right')
    img_charts['count_by_category'] = fig_to_base64(fig)

    # 💰 Total spend by category (Pie)
    cat_spend = paid.groupby('Category')['Amount'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(cat_spend, labels=cat_spend.index, autopct='%1.1f%%',
           colors=plt.cm.Paired(range(len(cat_spend))))
    ax.set_title("💰 Total Spend by Category")
    img_charts['pie_paid_category'] = fig_to_base64(fig)

    # 💸 Top subcategories by spend
    sub_spend = paid.groupby('SubCategory')['Amount'].sum().sort_values(ascending=False).head(15).reset_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=sub_spend, y='SubCategory', x='Amount', palette='viridis', ax=ax)
    ax.set_title("💸 Top 15 SubCategories by Paid Amount", fontsize=14, fontweight="bold")
    img_charts['sub_spend'] = fig_to_base64(fig)

    # 📆 Monthly trend (Paid vs Received)
    monthly = filtered_df.groupby(['Month', 'TransactionType'])['Amount'].sum().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly.plot(kind='bar', stacked=True, ax=ax, color=['#4B8BBE', '#F28C28'])
    ax.set_title("💰 Monthly Received vs Paid Amounts")
    plt.xticks(rotation=45)
    img_charts['monthly_paid_received'] = fig_to_base64(fig)

    # 🌳 Treemap — Spending by Category
    grouped = paid.groupby('Category')['Amount'].sum().reset_index().sort_values(by='Amount', ascending=False)
    grouped['Label'] = grouped.apply(lambda x: f"{x['Category']}\n₹{x['Amount']:.0f}", axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    squarify.plot(
        sizes=grouped['Amount'], label=grouped['Label'],
        color=sns.color_palette("pastel", len(grouped)), alpha=.9
    )
    plt.title("💸 Treemap: Spending by Category", fontsize=16, fontweight='bold')
    plt.axis('off')
    img_charts['treemap'] = fig_to_base64(fig)

    # ☁️ Word Cloud — Recipient Frequency
    rec_freq = Counter(paid['Recipient'].astype(str).tolist())
    wc = WordCloud(width=800, height=300, background_color='white').generate_from_frequencies(rec_freq)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('☁️ Word Cloud: Recipient Frequency', fontsize=14)
    plt.tight_layout()
    img_charts['wordcloud_recipients'] = fig_to_base64(fig)


    # ---------------------------------------------------------------
    # 🔹 Step 4: Heatmap — Spending Intensity (Weekday × Hour)
    # ---------------------------------------------------------------
    paid = filtered_df[filtered_df['TransactionType'] == 'Paid'].copy()
    if not paid.empty:
        paid['Date'] = pd.to_datetime(paid['Date'])
        paid['Hour'] = paid['Time'].apply(lambda x: int(str(x).split(':')[0]) if pd.notnull(x) else 0)
        paid['Weekday'] = paid['Date'].dt.day_name()

        pivot = paid.pivot_table(
            index='Weekday', columns='Hour', values='Amount',
            aggfunc='sum', fill_value=0
        )

        # Reorder weekdays
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = pivot.reindex(weekday_order)

        # Draw heatmap
        fig, ax = plt.subplots(figsize=(14, 5))
        sns.heatmap(
            pivot, cmap='YlGnBu', linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Total Amount (₹)'}, annot=False
        )
        plt.title('🕒 Spending Intensity by Weekday and Hour (Paid Transactions)', fontsize=14, pad=15)
        plt.xlabel('Hour of Day')
        plt.ylabel('Weekday')
        plt.xticks(rotation=0)
        plt.tight_layout()
        img_charts['heatmap_spending_intensity'] = fig_to_base64(fig)


    # ---------------------------------------------------------------
    # 🔹 Step 5: Smart Recommendations & ML Persona
    # ---------------------------------------------------------------
    recommendations = []
    latest_date = df['Date'].max()
    last_30_days = pd.to_datetime(end_date) - timedelta(days=30)
    recent_df = df[df['Date'] >= last_30_days]

    main_categories = ['Food', 'Travel', 'Shopping', 'Entertainment', 'Bills', 'Groceries']
    recent_main = recent_df[recent_df['Category'].isin(main_categories)]
    main_spend = recent_main.groupby('Category')['Amount'].sum()

    # 🧾 Spending pattern analysis
    if not main_spend.empty:
        top_main = main_spend.idxmax()
        val = main_spend.max()
        recommendations.append(f"🍽 You spent the most on **{top_main}** this week — ₹{int(val)}. Maybe track your {top_main.lower()} expenses!")

    if 'Food' in main_spend and main_spend['Food'] > 0.25 * main_spend.sum():
        recommendations.append("😋 Your food expenses are quite high this week. Ghar ka khana bhi maza deta hai!")

    if 'Travel' in main_spend and main_spend['Travel'] > 0.2 * main_spend.sum():
        recommendations.append("🛵 You travelled a lot recently. Try planning routes to save fuel or cab costs!")

    if 'Shopping' in main_spend and main_spend['Shopping'] > 0.15 * main_spend.sum():
        recommendations.append("🛍 You’ve been shopping quite a bit — maybe pause for a few days!")

    # ⏸ Inactive subcategories
    inactive = df.groupby('SubCategory')['Date'].max()
    inactive = inactive[inactive < (latest_date - timedelta(days=30))]
    for sub in inactive.index:
        recommendations.append(f"🔄 Haven’t spent on {sub} lately — still relevant?")

    # 📉 Weekly spending change
    weekly_spend = df.groupby('Week')['Amount'].sum()
    if len(weekly_spend) > 1:
        change = (weekly_spend.iloc[-1] - weekly_spend.iloc[-2]) / weekly_spend.iloc[-2]
        if change > 0.3:
            recommendations.append(f"💸 Spending increased by {int(change*100)}% this week — time to review your costs.")
        elif change < -0.2:
            recommendations.append("👏 Great! Spending reduced this week — keep it up!")

    # Generic finance tips
    recommendations += [
        "🌱 Invest in health, learning, and relationships — best returns ever!",
        "📆 Review subscriptions — cancel unused ones.",
        "🚀 Build an emergency fund for 3 months of expenses.",
        "💳 Use rewards/cashback on frequent spending areas."
    ]

    # ---------------------------------------------------------------
    # 🤖 ML Persona Detection (K-Means Clustering)
    # ---------------------------------------------------------------
    try:
        category_pivot = df.pivot_table(index='Week', columns='Category', values='Amount', aggfunc='sum', fill_value=0)
        scaler = StandardScaler()
        X = scaler.fit_transform(category_pivot)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        category_pivot['Cluster'] = kmeans.fit_predict(X)
        current_cluster = int(category_pivot.iloc[-1]['Cluster'])

        persona_labels = {
            0: ("Saver 🧘", "Balanced spending, low waste. Excellent control!"),
            1: ("Spender 💸", "High discretionary spending — monitor expenses!"),
            2: ("Adventurer 🏝", "Love travel and fun — save a bit more!"),
            3: ("Family Guy 👨‍👩‍👧‍👦", "Mostly essentials — stable pattern, enjoy leisure too!")
        }

        persona_name, persona_msg = persona_labels[current_cluster]
        recommendations.append(f"🧠 Based on your spending, you are a **{persona_name}**.")
        recommendations.append(persona_msg)
    except Exception:
        recommendations.append("⚙️ Could not determine persona this time due to limited data.")


    # ---------------------------------------------------------------
    # 🇮🇳 Hinglish-Styled Recommendations for Relatable UX
    # ---------------------------------------------------------------
    styled_recommendations = []
    endings = [
        "dekh le zarurat hai kya bhai 😅", "shayad ignore kar diya lagta hai 😬",
        "thoda check kar le bhai 👀", "lagta hai iss side dhyaan kam diya 😅",
        "agar important hai to thoda dhyaan de 💭", "kabhi kabhi balance bhi zaruri hota hai ⚖️",
        "budget check kar le — paisa bacha le bhai 💡", "dekh le kahin bhool to nahi gaya 😄"
    ]

    for rec in recommendations:
        if "Haven’t spent on" in rec:
            sub = rec.split("on ")[1].split(" lately")[0]
            msg = f"💬 🔄 {sub} pe kharcha nahi hua kuch time se — {random.choice(endings)}"
            styled_recommendations.append(msg)
        elif "food expenses" in rec:
            styled_recommendations.append("🍛 Khaane pe thoda zyada kharcha ho gaya bhai! Ghar ka khana bhi mast hota hai 😋")
        elif "travelled a lot" in rec:
            styled_recommendations.append("🛵 Bahut travel kar liya recently! Cab pooling ya metro try kar, paisa bachega 😉")
        elif "shopping" in rec:
            styled_recommendations.append("🛍 Shopping thodi zyada ho gayi lagta hai! Thoda brake le le bhai 😅")
        elif "Spender" in rec:
            styled_recommendations.append("💸 Thoda zyada kharch kar raha hai bhai! Budget bana aur control me rakh 😅")
        elif "Saver" in rec:
            styled_recommendations.append("🧘 Tu ek solid saver hai bhai! Smart aur balanced spending 👏")
        elif "Adventurer" in rec:
            styled_recommendations.append("🏝 Full adventure mode on hai! Travel mast hai, bas savings ka bhi khayal rakhna 😉")
        elif "Family Guy" in rec:
            styled_recommendations.append("👨‍👩‍👧‍👦 Family-focused spending accha hai, par thoda khud pe bhi kharcha kar bhai 😇")
        elif "emergency fund" in rec:
            styled_recommendations.append("🚀 Emergency fund banana mat bhool! 3 mahine ke kharche jitna to zaruri hai 💰")
        elif "subscription" in rec:
            styled_recommendations.append("📆 Subscription check kar le bhai — jo use nahi kar raha, wo cancel kar de!")
        else:
            styled_recommendations.append(rec)

    # Add motivational extras
    styled_recommendations += [
        "🌱 Health aur learning mein paisa lagana sabse accha investment hai 💖",
        "💳 Smart spending hi asli luxury hai — cashback aur rewards ka pura fayda uthao!",
        "🙏 Thoda savings apne future self ke naam bhi kar de — woh tujhe thank bolega!"
    ]


    # ---------------------------------------------------------------
    # 🎯 Step 6: Select Top 5 Diverse Recommendations
    # ---------------------------------------------------------------
    final_recommendations = []
    categories = ["Food", "Travel", "Shopping", "Savings", "Health", "Entertainment", "Family", "Investment"]
    used_tags = set()

    for rec in styled_recommendations:
        for cat in categories:
            if cat.lower() in rec.lower() and cat not in used_tags:
                final_recommendations.append(rec)
                used_tags.add(cat)
                break
        if len(final_recommendations) >= 5:
            break

    if len(final_recommendations) < 5:
        for rec in styled_recommendations:
            if rec not in final_recommendations:
                final_recommendations.append(rec)
            if len(final_recommendations) >= 5:
                break


    # ---------------------------------------------------------------
    # 🧭 Step 7: Render Final Dashboard Template
    # ---------------------------------------------------------------
    return render_template(
        'dashboard.html',
        start_date=start_date,
        end_date=end_date,
        category_chart=category_chart,
        type_chart=type_chart,
        img_charts=img_charts,
        recommendations=final_recommendations
    )


# ===============================================================
# 🚀 Run Flask App
# ===============================================================
if __name__ == "__main__":
    app.run(debug=True)
