import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# load
clf_model = pickle.load(open("clf_model.pkl", "rb"))
reg_model = pickle.load(open("reg_model.pkl", "rb"))

# Config
st.set_page_config(page_title="Student Prediction App", layout="wide")

st.title("Student Placement & Salary Prediction")
st.markdown("Masukkan data mahasiswa untuk memprediksi **placement** dan **estimasi salary**")

# SIDEBAR INFO
st.sidebar.header("Info")
st.sidebar.write("Aplikasi ini memprediksi kemungkinan mahasiswa mendapatkan pekerjaan dan estimasi gaji berdasarkan performa akademik dan skill.")

# form input
with st.form("prediction_form"):

    st.subheader("Input Data Mahasiswa")

    col1, col2 = st.columns(2)

    with col1:
        cgpa = st.slider("CGPA", 5.0, 10.0, 7.5)
        tenth = st.slider("Tenth Percentage", 50.0, 100.0, 75.0)
        twelfth = st.slider("Twelfth Percentage", 50.0, 100.0, 75.0)

        coding = st.slider("Coding Skill", 1, 5, 3)
        comm = st.slider("Communication Skill", 1, 5, 3)
        apt = st.slider("Aptitude Skill", 1, 5, 3)

        gender = st.selectbox("Gender", ["Male", "Female"])
        branch = st.selectbox("Branch", ["CSE", "ECE", "ME", "CE"])

    with col2:
        intern = st.slider("Internships", 0, 5, 1)
        projects = st.slider("Projects", 0, 10, 3)
        hackathons = st.slider("Hackathons", 0, 6, 2)

        study_hours = st.slider("Study Hours/Day", 0.0, 10.0, 4.0)
        attendance = st.slider("Attendance (%)", 40.0, 100.0, 75.0)

        backlogs = st.slider("Backlogs", 0, 5, 0)
        certifications = st.slider("Certifications", 0, 10, 2)

        sleep = st.slider("Sleep Hours", 4.0, 9.0, 7.0)
        stress = st.slider("Stress Level", 1, 10, 5)

        part_time = st.selectbox("Part Time Job", ["Yes", "No"])
        income = st.selectbox("Family Income", ["Low", "Medium", "High"])
        city = st.selectbox("City Tier", ["Tier1", "Tier2", "Tier3"])

        internet = st.selectbox("Internet Access", ["Yes", "No"])
        extra = st.selectbox("Extracurricular", ["Yes", "No"])

    submitted = st.form_submit_button("Predict")

# process & predict
if submitted:

    # feature engineering
    total_skills = coding + comm + apt

    academic_score = (
        cgpa +
        tenth / 10 +
        twelfth / 10
    )

    experience_score = intern + projects + hackathons

    performance_index = (
        academic_score +
        total_skills +
        experience_score
    )

    cgpa_skill = cgpa * coding

    # Dataframe input
    input_data = pd.DataFrame([{
        "gender": gender,
        "branch": branch,
        "backlogs": backlogs,
        "certifications_count": certifications,
        "sleep_hours": sleep,
        "stress_level": stress,
        "part_time_job": part_time,
        "family_income_level": income,
        "city_tier": city,
        "internet_access": internet,
        "extracurricular_involvement": extra,

        "cgpa": cgpa,
        "tenth_percentage": tenth,
        "twelfth_percentage": twelfth,
        "study_hours_per_day": study_hours,
        "attendance_percentage": attendance,

        "coding_skill_rating": coding,
        "communication_skill_rating": comm,
        "aptitude_skill_rating": apt,

        "internships_completed": intern,
        "projects_completed": projects,
        "hackathons_participated": hackathons,

        "total_skills": total_skills,
        "academic_score": academic_score,
        "experience_score": experience_score,
        "performance_index": performance_index,
        "cgpa_skill": cgpa_skill
    }])

    # prediction
    placement = clf_model.predict(input_data)[0]

    col1, col2 = st.columns(2)

    # result
    with col1:
        st.subheader("📊 Prediction Result")

        if placement == 1:
            salary = reg_model.predict(input_data)[0]
            st.success(f"Placed!")
            st.write(f"Estimated Salary: **{salary:.2f} LPA**")
        else:
            st.error("Not Placed")

    # VISUALIZATION
    with col2:
        st.subheader("📈 Performance Visualization")

        fig, ax = plt.subplots()

        labels = ["Skills", "Academic", "Experience"]
        values = [total_skills, academic_score, experience_score]

        ax.bar(labels, values)
        ax.set_title("Performance Breakdown")

        st.pyplot(fig)

    st.subheader("Input Data Summary")
    st.dataframe(input_data)