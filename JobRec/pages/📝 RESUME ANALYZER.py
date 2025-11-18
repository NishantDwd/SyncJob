# import streamlit as st
# import pandas as pd
# import base64,random
# import time,datetime
# from pyresparser import ResumeParser
# import io,random
# from streamlit_tags import st_tags
# from PIL import Image
# import pymongo
# import plotly.express as px
# from JobRecommendation.side_logo import add_logo
# from JobRecommendation.sidebar import sidebar
# from JobRecommendation.courses import ds_course,web_course,android_course,ios_course,uiux_course,ds_keyword,web_keyword,android_keyword,ios_keyword,uiux_keyword
# from JobRecommendation import utils ,MongoDB_function
# dataBase = "Job-Recomendation"
# collection = "Resume_from_RESUME_ANALYZER"
# st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="RESUME ANALYZER")


# add_logo()
# sidebar()


# def course_recommender(course_list):
#     st.subheader("*Courses & Certificates🎓 Recommendations*")
#     c = 0
#     rec_course = []
#     no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 4)
#     random.shuffle(course_list)
#     for c_name, c_link in course_list:
#         c += 1
#         st.markdown(f"({c}) [{c_name}]({c_link})")
#         rec_course.append(c_name)
#         if c == no_of_reco:
#             break
#     return rec_course


# def run():
#         st.title("Resume Analyser")


#         pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
#         if pdf_file is not None:
#             count_=0

#             encoded_pdf=utils.pdf_to_base64(pdf_file)
#             # embed_code=utils.show_pdf(encoded_pdf)
#             # st.markdown(embed_code, unsafe_allow_html=True)
#             resume_data = ResumeParser(pdf_file).get_extracted_data()

#             resume_data["pdf_to_base64"]=encoded_pdf
            
#             #resume_store(resume_data)
#             if resume_data:
#                 ## Get the whole resume data
#                 resume_text = utils.pdf_reader(pdf_file)

#                 st.header("Resume Analysis")

#                 try:
#                     st.success("Hello "+ resume_data['name'])
#                     st.subheader("Your Basic info")
#                     st.text('Name: '+resume_data['name'])
#                     st.text('Email: ' + resume_data['email'])
#                     st.text('Contact: ' + resume_data['mobile_number'])
#                     st.text('Resume pages: '+str(resume_data['no_of_pages']))
#                 except:
#                     pass
#                 cand_level = ''
#                 if resume_data['no_of_pages'] == 1:
#                     cand_level = "Fresher"
#                     st.markdown( '''<h4 style='text-align: left; color: #d73b5c;'>You are looking Fresher.</h4>''',unsafe_allow_html=True)
#                 elif resume_data['no_of_pages'] == 2:
#                     cand_level = "Intermediate"
#                     st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',unsafe_allow_html=True)
#                 elif resume_data['no_of_pages'] >=3:
#                     cand_level = "Experienced"
#                     st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',unsafe_allow_html=True)
                

#                 st.subheader("**Skills Recommendation💡**")
#                 ## Skill shows
#                 keywords = st_tags(label='### Skills that you have',
#                 text='See our skills recommendation',
#                     value=resume_data['skills'],key = '1')

#                 recommended_skills = []
#                 reco_field = ''
#                 rec_course = ''
#                 ## Courses recommendation
#                 for i in resume_data['skills']:
#                     ## Data science recommendation
#                     if i.lower() in ds_keyword:
#                         print(i.lower())
#                         reco_field = 'Data Science'
#                         st.success("** Our analysis says you are looking for Data Science Jobs.**")
#                         recommended_skills = ['Data Visualization','Predictive Analysis','Statistical Modeling','Data Mining','Clustering & Classification','Data Analytics','Quantitative Analysis','Web Scraping','ML Algorithms','Keras','Pytorch','Probability','Scikit-learn','Tensorflow',"Flask",'Streamlit']
#                         recommended_keywords = st_tags(label='### Recommended skills for you.',
#                         text='Recommended skills generated from System',value=recommended_skills,key = '2')
#                         st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',unsafe_allow_html=True)
#                         rec_course = course_recommender(ds_course)
#                         break

#                     ## Web development recommendation
#                     elif i.lower() in web_keyword:
#                         print(i.lower())
#                         reco_field = 'Web Development'
#                         st.success("** Our analysis says you are looking for Web Development Jobs **")
#                         recommended_skills = ['React','Django','Node JS','React JS','php','laravel','Magento','wordpress','Javascript','Angular JS','c#','Flask','SDK']
#                         recommended_keywords = st_tags(label='### Recommended skills for you.',
#                         text='Recommended skills generated from System',value=recommended_skills,key = '3')
#                         st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',unsafe_allow_html=True)
#                         rec_course = course_recommender(web_course)
#                         break

#                     ## Android App Development
#                     elif i.lower() in android_keyword:
#                         print(i.lower())
#                         reco_field = 'Android Development'
#                         st.success("** Our analysis says you are looking for Android App Development Jobs **")
#                         recommended_skills = ['Android','Android development','Flutter','Kotlin','XML','Java','Kivy','GIT','SDK','SQLite']
#                         recommended_keywords = st_tags(label='### Recommended skills for you.',
#                         text='Recommended skills generated from System',value=recommended_skills,key = '4')
#                         st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',unsafe_allow_html=True)
#                         rec_course = course_recommender(android_course)
#                         break

#                     ## IOS App Development
#                     elif i.lower() in ios_keyword:
#                         print(i.lower())
#                         reco_field = 'IOS Development'
#                         st.success("** Our analysis says you are looking for IOS App Development Jobs **")
#                         recommended_skills = ['IOS','IOS Development','Swift','Cocoa','Cocoa Touch','Xcode','Objective-C','SQLite','Plist','StoreKit',"UI-Kit",'AV Foundation','Auto-Layout']
#                         recommended_keywords = st_tags(label='### Recommended skills for you.',
#                         text='Recommended skills generated from System',value=recommended_skills,key = '5')
#                         st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',unsafe_allow_html=True)
#                         rec_course = course_recommender(ios_course)
#                         break

#                     ## Ui-UX Recommendation
#                     elif i.lower() in uiux_keyword:
#                         print(i.lower())
#                         reco_field = 'UI-UX Development'
#                         st.success("** Our analysis says you are looking for UI-UX Development Jobs **")
#                         recommended_skills = ['ux','adobe xd','figma','zeplin','balsamiq','ui','prototyping','wireframes','storyframes','adobe photoshop','photoshop','editing','adobe illustrator','illustrator','adobe after effects','after effects','adobe premier pro','premier pro','adobe indesign','indesign','wireframe','solid','grasp','user research','user experience']
#                         recommended_keywords = st_tags(label='### Recommended skills for you.',
#                         text='Recommended skills generated from System',value=recommended_skills,key = '6')
#                         st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',unsafe_allow_html=True)
#                         rec_course = course_recommender(uiux_course)
#                         break


#                 # inserting data into mongodb  
#                 timestamp = utils.generateUniqueFileName()
#                 save={timestamp:resume_data}
#                 if count_==0:
#                     count_=1
#                     MongoDB_function.resume_store(save,dataBase,collection)

#                 ### Resume writing recommendation
#                 st.subheader("**Resume Tips & Ideas💡**")
#                 resume_score = 0
#                 if 'Objective' in resume_text:
#                     resume_score = resume_score+20
#                     st.markdown('''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Objective</h4>''',unsafe_allow_html=True)
#                 else:
#                     st.markdown('''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add your career objective, it will give your career intension to the Recruiters.</h4>''',unsafe_allow_html=True)

#                 if 'Declaration'  in resume_text:
#                     resume_score = resume_score + 20
#                     st.markdown('''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Delcaration✍/h4>''',unsafe_allow_html=True)
#                 else:
#                     st.markdown('''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add Declaration✍. It will give the assurance that everything written on your resume is true and fully acknowledged by you</h4>''',unsafe_allow_html=True)

#                 if 'Hobbies' or 'Interests'in resume_text:
#                     resume_score = resume_score + 20
#                     st.markdown('''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Hobbies⚽</h4>''',unsafe_allow_html=True)
#                 else:
#                     st.markdown('''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add Hobbies⚽. It will show your persnality to the Recruiters and give the assurance that you are fit for this role or not.</h4>''',unsafe_allow_html=True)

#                 if 'Achievements' in resume_text:
#                     resume_score = resume_score + 20
#                     st.markdown('''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Achievements🏅 </h4>''',unsafe_allow_html=True)
#                 else:
#                     st.markdown('''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add Achievements🏅. It will show that you are capable for the required position.</h4>''',unsafe_allow_html=True)

#                 if 'Projects' in resume_text:
#                     resume_score = resume_score + 20
#                     st.markdown('''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects👨‍💻 </h4>''',unsafe_allow_html=True)
#                 else:
#                     st.markdown('''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add Projects👨‍💻. It will show that you have done work related the required position or not.</h4>''',unsafe_allow_html=True)

#                 st.subheader("**Resume Score📝**")
#                 st.markdown(
#                     """
#                     <style>
#                         .stProgress > div > div > div > div {
#                             background-color: #d73b5c;
#                         }
#                     </style>""",
#                     unsafe_allow_html=True,
#                 )
#                 my_bar = st.progress(0)
#                 score = 0
#                 for percent_complete in range(resume_score):
#                     score +=1
#                     time.sleep(0.1)
#                     my_bar.progress(percent_complete + 1)
#                 st.success('** Your Resume Writing Score: ' + str(score)+'**')
#                 st.warning("** Note: This score is calculated based on the content that you have added in your Resume. **")
#                 st.balloons()

                

#             else:
#                 st.error("Wrong ID & Password Provided")

#             st.balloons()           




# run()


import streamlit as st
import pandas as pd
import base64
import random
import time
import datetime
from pyresparser import ResumeParser
import io
from streamlit_tags import st_tags
from PIL import Image
import pymongo
import plotly.express as px

from JobRecommendation.side_logo import add_logo
from JobRecommendation.sidebar import sidebar
from JobRecommendation.courses import (
    ds_course, web_course, android_course, ios_course, uiux_course,
    ds_keyword, web_keyword, android_keyword, ios_keyword, uiux_keyword
)
from JobRecommendation import utils, MongoDB_function

# -------------------------
# Config
# -------------------------
dataBase = "Job-Recomendation"
collection = "Resume_from_RESUME_ANALYZER"
st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="RESUME ANALYZER")

add_logo()
sidebar()


def course_recommender(course_list):
    st.subheader("*Courses & Certificates🎓 Recommendations*")
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 4)

    course_copy = course_list.copy()
    random.shuffle(course_copy)
    for idx, (c_name, c_link) in enumerate(course_copy, start=1):
        st.markdown(f"({idx}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if idx == no_of_reco:
            break
    return rec_course


def detect_domain_from_skills(skills):
    """
    Count matches across domain keyword lists and return best domain and score map.
    Keywords lists are expected to contain lowercase strings (but we lowercase anyway).
    """
    skill_set = {s.strip().lower() for s in skills if isinstance(s, str) and s.strip()}
    domain_scores = {
        "Data Science": sum(1 for s in skill_set if s in {k.lower() for k in ds_keyword}),
        "Web Development": sum(1 for s in skill_set if s in {k.lower() for k in web_keyword}),
        "Android Development": sum(1 for s in skill_set if s in {k.lower() for k in android_keyword}),
        "IOS Development": sum(1 for s in skill_set if s in {k.lower() for k in ios_keyword}),
        "UI-UX Development": sum(1 for s in skill_set if s in {k.lower() for k in uiux_keyword}),
    }
    # If all zeros, return None to allow fallback
    if all(v == 0 for v in domain_scores.values()):
        return None, domain_scores
    best = max(domain_scores, key=domain_scores.get)
    return best, domain_scores


def domain_recommendations(reco_field):
    """
    Return recommended skills list and course list corresponding to reco_field.
    """
    if reco_field == "Data Science":
        return (
            ['Data Visualization', 'Predictive Analysis', 'Statistical Modeling', 'Data Mining',
             'Clustering & Classification', 'Data Analytics', 'Quantitative Analysis',
             'Web Scraping', 'ML Algorithms', 'Keras', 'Pytorch', 'Probability',
             'Scikit-learn', 'Tensorflow', 'Flask', 'Streamlit'],
            ds_course
        )
    if reco_field == "Web Development":
        return (
            ['React', 'Django', 'Node JS', 'React JS', 'php', 'laravel', 'Magento',
             'wordpress', 'Javascript', 'Angular JS', 'c#', 'Flask', 'SDK'],
            web_course
        )
    if reco_field == "Android Development":
        return (
            ['Android', 'Android development', 'Flutter', 'Kotlin', 'XML', 'Java', 'Kivy',
             'GIT', 'SDK', 'SQLite'],
            android_course
        )
    if reco_field == "IOS Development":
        return (
            ['IOS', 'IOS Development', 'Swift', 'Cocoa', 'Cocoa Touch', 'Xcode', 'Objective-C',
             'SQLite', 'Plist', 'StoreKit', 'UI-Kit', 'AV Foundation', 'Auto-Layout'],
            ios_course
        )
    if reco_field == "UI-UX Development":
        return (
            ['ux', 'adobe xd', 'figma', 'zeplin', 'balsamiq', 'ui', 'prototyping', 'wireframes',
             'storyframes', 'adobe photoshop', 'photoshop', 'editing', 'adobe illustrator',
             'illustrator', 'adobe after effects', 'after effects', 'adobe premier pro',
             'premier pro', 'adobe indesign', 'indesign', 'wireframe', 'user research',
             'user experience'],
            uiux_course
        )
    # Default
    return ([], [])


def safe_get_resume_text(pdf_file):
    """
    Try to read text using utils.pdf_reader fallback safely.
    """
    try:
        return utils.pdf_reader(pdf_file) or ""
    except Exception:
        try:
            # try simple fallback: read bytes and decode (not robust, but better than crash)
            pdf_file.seek(0)
            raw = pdf_file.read()
            if isinstance(raw, bytes):
                return str(raw[:1000])
            return ""
        except Exception:
            return ""


def run():
    st.title("Resume Analyser")

    pdf_file = st.file_uploader("Choose your Resume (PDF)", type=["pdf"])
    if pdf_file is None:
        st.info("Upload a PDF resume to analyze.")
        return

    # single-run counter to avoid duplicate DB inserts on rerun
    count_ = 0

    # base64 encode for embedding / storing
    try:
        encoded_pdf = utils.pdf_to_base64(pdf_file)
    except Exception:
        encoded_pdf = None

    # Attempt to parse resume data with pyresparser safely
    try:
        resume_data = ResumeParser(pdf_file).get_extracted_data()
    except Exception as e:
        resume_data = {}
        st.warning("Resume parsing via pyresparser failed or produced incomplete data. Falling back to text-based heuristics.")

    if resume_data is None:
        resume_data = {}

    # Ensure keys exist
    resume_data.setdefault("name", "Not found")
    resume_data.setdefault("email", "Not found")
    resume_data.setdefault("mobile_number", "Not found")
    resume_data.setdefault("skills", [])
    resume_data.setdefault("no_of_pages", 1)
    if encoded_pdf:
        resume_data["pdf_to_base64"] = encoded_pdf

    # read resume full text (lowercased for checks)
    resume_text = safe_get_resume_text(pdf_file)
    resume_text_lower = resume_text.lower()

    st.header("Resume Analysis")

    # Basic info block (guarded)
    try:
        if resume_data.get("name"):
            st.success("Hello " + str(resume_data.get("name")))
        st.subheader("Your Basic info")
        st.text('Name: ' + str(resume_data.get("name", "Not found")))
        st.text('Email: ' + str(resume_data.get("email", "Not found")))
        st.text('Contact: ' + str(resume_data.get("mobile_number", "Not found")))
        st.text('Resume pages: ' + str(resume_data.get("no_of_pages", 1)))
    except Exception:
        st.write("Some basic info could not be displayed.")

    # Candidate level
    cand_level = ''
    try:
        pages = int(resume_data.get('no_of_pages', 1))
    except Exception:
        pages = 1

    if pages == 1:
        cand_level = "Fresher"
        st.markdown("<h4 style='text-align: left; color: #d73b5c;'>You are looking Fresher.</h4>", unsafe_allow_html=True)
    elif pages == 2:
        cand_level = "Intermediate"
        st.markdown("<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>", unsafe_allow_html=True)
    elif pages >= 3:
        cand_level = "Experienced"
        st.markdown("<h4 style='text-align: left; color: #fba171;'>You are at experience level!</h4>", unsafe_allow_html=True)

    st.subheader("**Skills Recommendation💡**")

    # show skills (pyresparser might produce list or comma string)
    raw_skills = resume_data.get('skills', [])
    if isinstance(raw_skills, str):
        # try to split by comma
        skills_list = [s.strip() for s in raw_skills.split(',') if s.strip()]
    elif isinstance(raw_skills, (list, set, tuple)):
        skills_list = list(raw_skills)
    else:
        skills_list = []

    # interactive tag input
    keywords = st_tags(label='### Skills that you have',
                       text='See our skills recommendation',
                       value=skills_list, key='1')

    # Domain detection - from skills first
    reco_field, domain_scores = detect_domain_from_skills(keywords)

    # fallback: if none matched, try to search domain-specific keywords in resume text
    if reco_field is None:
        resume_lower = resume_text_lower
        # simple heuristics: count occurrences of domain keywords in text
        domain_scores_text = {
            "Data Science": sum(resume_lower.count(w.lower()) for w in ds_keyword),
            "Web Development": sum(resume_lower.count(w.lower()) for w in web_keyword),
            "Android Development": sum(resume_lower.count(w.lower()) for w in android_keyword),
            "IOS Development": sum(resume_lower.count(w.lower()) for w in ios_keyword),
            "UI-UX Development": sum(resume_lower.count(w.lower()) for w in uiux_keyword),
        }
        if any(v > 0 for v in domain_scores_text.values()):
            reco_field = max(domain_scores_text, key=domain_scores_text.get)
            domain_scores = domain_scores_text

    # final fallback: if still None, mark as General
    if reco_field is None:
        reco_field = "General / Other"

    st.success(f"**Detected Career Domain → {reco_field}**")

    # Get recommended skills and course list
    recommended_skills, course_list = domain_recommendations(reco_field)

    if recommended_skills:
        st.markdown("### Recommended skills for you (add to resume):")
        recommended_keywords = st_tags(label='',
                                      text='Recommended skills generated from System',
                                      value=recommended_skills, key='2')
        st.markdown("<h4 style='text-align: left; color: #1ed760;'>Adding these skills to your resume will boost🚀 the chances of getting a Job💼</h4>", unsafe_allow_html=True)

    # Show course recommendations if we have a course list
    if course_list:
        rec_course = course_recommender(course_list)

    # Inserting data into mongodb (safe)
    try:
        timestamp = utils.generateUniqueFileName()
        save = {timestamp: resume_data}
        if count_ == 0:
            count_ = 1
            MongoDB_function.resume_store(save, dataBase, collection)
    except Exception as e:
        st.info("Could not store resume to DB: " + str(e))

    ### Resume writing recommendation & scoring
    st.subheader("**Resume Tips & Ideas💡**")
    resume_score = 0
    # We'll check for presence of sections/keywords in lowercased resume text
    # Each found section adds 20 points (Objective, Declaration, Hobbies/Interests, Achievements, Projects)
    checks = [
        ("objective", "[+] Objective section found ✅", "[-] Add an Objective section."),
        ("declaration", "[+] Declaration section found ✅", "[-] Add a Declaration section."),
        # Hobbies OR Interests
        (("hobbies", "interests"), "[+] Hobbies/Interests section found ✅", "[-] Mention Hobbies/Interests."),
        ("achievement", "[+] Achievements section found ✅", "[-] Add Achievements section."),
        ("project", "[+] Projects section found ✅", "[-] Add Projects section."),
    ]

    for key, success_msg, fail_msg in checks:
        found = False
        if isinstance(key, tuple):
            # any of these
            found = any(k in resume_text_lower for k in key)
        else:
            found = (key in resume_text_lower)
        if found:
            resume_score += 20
            st.markdown(f"<h4 style='text-align: left; color: #1ed760;'>{success_msg}</h4>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h4 style='text-align: left; color: #fabc10;'>{fail_msg}</h4>", unsafe_allow_html=True)

    # Ensure score between 0 and 100
    resume_score = max(0, min(100, resume_score))

    st.subheader("**Resume Score📝**")
    st.markdown("""
        <style>
            .stProgress > div > div > div > div { background-color: #d73b5c; }
        </style>""", unsafe_allow_html=True)

    # Faster progress animation (deterministic, no long sleeps)
    my_bar = st.progress(0)
    for percent_complete in range(resume_score + 1):  # include resume_score
        my_bar.progress(percent_complete)
        time.sleep(0.01)  # short sleep so UI shows change quickly
    st.success(f'** Your Resume Writing Score: {resume_score} **')
    st.warning("** Note: This score is calculated based on the presence of sections (Objective, Declaration, Hobbies/Interests, Achievements, Projects). Improve your resume content to increase this score. **")

    # Optional: show domain score breakdown (nice info)
    try:
        st.subheader("Domain match counts")
        df = pd.DataFrame(list(domain_scores.items()), columns=["Domain", "Match Count"]).sort_values(by="Match Count", ascending=False)
        st.table(df)
    except Exception:
        pass

    # One final balloon
    st.balloons()


if __name__ == "__main__":
    run()

