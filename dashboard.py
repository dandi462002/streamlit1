import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib

sns.set(style='dark')

def load_data():
    return pd.read_csv("attrition_dashboard_data.csv")  

def load_model():
    return joblib.load("model_attrition.pkl")

df = load_data()
model = load_model()

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
nominal_cols = df.select_dtypes(include=['object']).columns.tolist()

ordinal_cols = [col for col in numeric_cols if df[col].nunique() <= 10]

exclude_cols = ['EmployeeId', 'EmployeeCount', 'StandardHours']
ordinal_cols = [col for col in ordinal_cols if col not in exclude_cols]

numeric_features = [col for col in numeric_cols if col not in ordinal_cols]

ordinal_label_map = {
    "WorkLifeBalance": {
        1: "1 - Low", 2: "2 - Good", 3: "3 - Excellent", 4: "4 - Outstanding"
    },
    "PerformanceRating": {
        1: "1 - Low", 2: "2 - Good", 3: "3 - Excellent", 4: "4 - Outstanding"
    },
    "RelationshipSatisfaction": {
        1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"
    },
    "JobSatisfaction": {
        1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"
    },
    "Education": {
        1: "1 - Below College", 2: "2 - College", 3: "3 - Bachelor", 4: "4 - Master", 5: "5 - Doctor"
    },
    "EnvironmentSatisfaction": {
        1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"
    },
    "JobInvolvement": {
        1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"
    }
}

menu = st.sidebar.radio("Menu", [
    "Data Overview",
    "Feature Importance",
    "Komparasi Fitur terhadap Attrition",
    "Predict Attrition (Inference)",
    "Insights & Recommendations"
])

#  MENU 1: Data Overview
if menu == "Data Overview":
    st.title("📊 Data Overview")
    st.write("Berikut adalah gambaran umum dari dataset yang digunakan:")
    st.dataframe(df)
    st.write("Jumlah data:", df.shape[0])
    st.write("Jumlah fitur:", df.shape[1])
    feature_desc = {
        "Age": "Age of the employee",
        "Attrition": "Did the employee attrition? (0=no, 1=yes)",
        "BusinessTravel": "Travel commitments for the job",
        "DailyRate": "Daily salary",
        "Department": "Employee Department",
        "DistanceFromHome": "Distance from work to home (in km)",
        "Education": "1-Below College, 2-College, 3-Bachelor, 4-Master,5-Doctor",
        "EducationField": "Field of Education",
        "EnvironmentSatisfaction": "1-Low, 2-Medium, 3-High, 4-Very High",
        "Gender": "Employee's gender",
        "HourlyRate": "Hourly salary",
        "JobInvolvement": "1-Low, 2-Medium, 3-High, 4-Very High",
        "JobLevel": "Level of job (1 to 5)",
        "JobRole": "Job Roles",
        "JobSatisfaction": "1-Low, 2-Medium, 3-High, 4-Very High",
        "MaritalStatus": "Marital Status",
        "MonthlyIncome": "Monthly salary",
        "MonthlyRate": "Monthly rate",
        "NumCompaniesWorked": "Number of companies worked at",
        "OverTime": "Overtime?",
        "PercentSalaryHike": "The percentage increase in salary last year",
        "PerformanceRating": "1-Low, 2-Good, 3-Excellent, 4-Outstanding",
        "RelationshipSatisfaction": "1-Low, 2-Medium, 3-High, 4-Very High",
        "StandardHours": "Standard Hours",
        "StockOptionLevel": "Stock Option Level",
        "TotalWorkingYears": "Total years worked",
        "TrainingTimesLastYear": "Number of training attended last year",
        "WorkLifeBalance": "1-Low, 2-Good, 3-Excellent, 4-Outstanding",
        "YearsAtCompany": "Years at Company",
        "YearsInCurrentRole": "Years in the current role",
        "YearsSinceLastPromotion": "Years since the last promotion",
        "YearsWithCurrManager": "Years with the current manager"
    }

    # Convert ke DataFrame
    desc_df = pd.DataFrame(list(feature_desc.items()), columns=["Feature", "Description"])

    st.subheader("📋 Deskripsi Kolom Dataset")
    st.dataframe(desc_df, use_container_width=True)

    # Visualisasi distribusi target
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='Attrition', data=df, ax=ax)
    ax.set_title('Distribusi Target: Attrition')
    st.pyplot(fig)

#  MENU 2: Feature Importance 
if menu == "Feature Importance":
    st.title("🔍 Feature Importance dari Model")
    st.write("Berikut adalah fitur-fitur yang paling berpengaruh terhadap prediksi Attrition:")
    classifier = model.named_steps['classifier']
    feature_names = model.named_steps['preprocessing'].get_feature_names_out()
    if hasattr(classifier, 'coef_'):
        importances = classifier.coef_[0]

        imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        imp_df['Abs_Importance'] = imp_df['Importance'].abs()
        imp_df = imp_df.sort_values(by='Abs_Importance', ascending=False)

        st.dataframe(imp_df, use_container_width=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=imp_df.head(15),
            x='Importance',
            y='Feature',
            ax=ax
        )
        ax.set_title('Top 15 Important Features')
        st.pyplot(fig)

    else:
        st.warning("Model tidak mendukung feature importance.")

#  MENU 3: Compare Features with Attrition 
elif menu == "Komparasi Fitur terhadap Attrition":
    st.title("📊 Komparasi Fitur terhadap Attrition")
    selected_feature = st.selectbox("Pilih fitur untuk dibandingkan dengan Attrition:", df.columns.drop('Attrition'))
    fig, ax = plt.subplots(figsize=(9, 5))
    if selected_feature in nominal_cols:
        sns.countplot(x=selected_feature, hue='Attrition', data=df, ax=ax)
        ax.set_title(f"{selected_feature} (Nominal) vs Attrition")

    elif selected_feature in ordinal_cols:
        df_viz = df.copy()
        if selected_feature in ordinal_label_map:
            df_viz[selected_feature] = df_viz[selected_feature].map(ordinal_label_map[selected_feature])

        sns.countplot(x=selected_feature, hue='Attrition', data=df_viz, ax=ax)
        ax.set_title(f"{selected_feature} (Ordinal) vs Attrition")

    elif selected_feature in numeric_features:
        sns.boxplot(x='Attrition', y=selected_feature, data=df, ax=ax)
        ax.set_title(f"{selected_feature} (Numerik) vs Attrition")

    else:
        st.warning("Tipe fitur tidak dikenali untuk visualisasi.")
    st.pyplot(fig)

#  MENU 4: Inference / Predict Attrition 
elif menu == "Predict Attrition (Inference)":
    st.title("🤖 Prediksi Attrition dari Input Data")
    st.markdown("Silakan masukkan data karyawan berikut:")

    user_input = {}
    for col in df.columns.drop('Attrition'):
        if df[col].dtype == 'object':
            user_input[col] = st.selectbox(col, options=sorted(df[col].dropna().unique()))
        elif col in ordinal_cols:
            min_val = int(df[col].min())
            max_val = int(df[col].max())
            user_input[col] = st.selectbox(col, options=list(range(min_val, max_val + 1)))
        else:
            user_input[col] = st.number_input(col, value=float(df[col].mean()))



    if st.button("Prediksi"):
        input_df = pd.DataFrame([user_input])
        input_df = input_df[df.drop('Attrition', axis=1).columns]  

        try:
            prediction = model.predict(input_df)[0]
            result = "⚠️ Karyawan berisiko resign." if prediction == 1 else "✅ Karyawan diprediksi tidak resign."
            st.success(result)
        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")

# === MENU 5: Insight & Recommendations ===
elif menu == "Insights & Recommendations":
    st.title("💡 Insight dan Rekomendasi Strategis")
    
    st.subheader("📌 Temuan Utama (Model Insights)")
    st.markdown("""
    Berdasarkan analisis koefisien model terhadap data Attrition, berikut adalah faktor pendorong utama:
    * **Pemicu Keluar Terkuat:** Budaya **Lembur (OverTime)** dan frekuensi **Perjalanan Dinas** yang tinggi.
    * **Kelompok Rentan:** Karyawan dengan status **Single** serta posisi teknis seperti **Laboratory Technician** dan **Sales Representative**.
    * **Faktor Retensi:** Jabatan senior (Manager/Director) dan total masa kerja yang panjang memiliki tingkat loyalitas tertinggi.
    """)

    st.divider()

    st.subheader("🚀 Rekomendasi Tindakan untuk HR")
    st.markdown("""
    1.  **Evaluasi Beban Kerja & Lembur:** Segera audit departemen dengan jam lembur tinggi. Pertimbangkan penambahan staf atau efisiensi alur kerja untuk mengurangi ketergantungan pada *Overtime*.
    
    2.  **Manajemen Perjalanan Dinas:** Berikan periode istirahat atau rotasi tugas bagi karyawan yang sering bepergian (*Travel Frequently*) untuk mencegah kelelahan fisik dan mental (*burnout*).
    
    3.  **Program Retensi Karyawan Baru:** Fokuskan pendampingan (mentorship) pada karyawan dengan masa kerja di bawah 3 tahun, karena data menunjukkan risiko *attrition* menurun seiring bertambahnya senioritas.
    
    4.  **Penyesuaian Jalur Karier Teknis:** Tinjau kembali kompensasi dan jenjang karier untuk peran spesifik seperti *Laboratory Technician* agar lebih kompetitif dibandingkan tawaran eksternal.
    
    5.  **Engagement Karyawan Muda/Single:** Ciptakan lingkungan kerja yang inklusif dan program *Work-Life Balance* yang menarik bagi segmen karyawan muda untuk meningkatkan rasa memiliki (*sense of belonging*).

    ---
    _Dashboard ini dikembangkan untuk mendukung pengambilan keputusan HR yang berbasis data (Data-Driven Decision Making)._
    """)