import numpy
import sys
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.preprocessing import OrdinalEncoder
import warnings
numpy.set_printoptions(threshold=sys.maxsize) #print entire numpy arrays

def preprocess_student_data(student_data):
    #student_data = student_data.drop_duplicates(subset=['SPRIDEN_PIDM'])
    student_data.reset_index(inplace=True, drop=True)

    return student_data

def create_features_matrix_and_target_vector_for_rf_model(student_data, training_course):
    indices_of_students_in_training_course = []

    is_in_course = np.zeros(shape=(len(student_data), 1))

    for i in range(0, len(student_data)):
        if str(student_data.iloc[i]["SPRIDEN_PIDM"]) in training_course["SPRIDEN_PIDM"].to_string():
            indices_of_students_in_training_course.append(i)
            is_in_course[i] = 1

    student_data["Enrolled in Course Next Year"] = is_in_course

    students_in_training_course = []
    for i in indices_of_students_in_training_course:
        students_in_training_course.append(student_data.iloc[i])

    target_vector = np.ones(shape=(len(students_in_training_course), 1))

    features_matrix = pd.DataFrame(students_in_training_course)
    features_matrix["Enrolled in Course Next Year"] = target_vector

    features_matrix = features_matrix._append(student_data.sample(n=len(features_matrix) * 1)) # make so only non-students are sampled?

    return features_matrix.drop(columns="Enrolled in Course Next Year"), np.array(features_matrix["Enrolled in Course Next Year"])


def create_rf_model_for_course(all_student_data, course_subject, course_number):
    fall_2021_students_df = all_student_data.loc[
        (all_student_data["Academic Term"] == "Fall") & (all_student_data["Academic Year"] == "2020-2021")]

    fall_2021_students_df = preprocess_student_data(fall_2021_students_df)

    target_vector = fall_2021_students_df[course_subject + "_" + course_number]

    rf_model = RandomForestClassifier()
    fall_2021_students_df.drop(columns=fall_2021_students_df.iloc[:, :6], inplace=True)
    fall_2021_students_df = fall_2021_students_df.iloc[:, :-6]
    rf_model.fit(fall_2021_students_df, target_vector)

    fall_2021_students_df = all_student_data.loc[
        (all_student_data["Academic Term"] == "Fall") & (all_student_data["Academic Year"] == "2021-2022")]

    fall_2021_students_df = preprocess_student_data(fall_2021_students_df)


    target_vector = np.zeros(shape=(len(fall_2021_students_df), 1))

    # for i in range(0, len(fall_2021_students_df)):
    #     if str(fall_2021_students_df.iloc[i]["SPRIDEN_PIDM"]) in training_course["SPRIDEN_PIDM"].to_string():
    #         target_vector[i] = 1

    y_pred = rf_model.predict(fall_2021_students_df.drop(columns=["Academic Year", "Academic Term", "SPRIDEN_PIDM"]))

    cm = confusion_matrix(target_vector, y_pred)

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)

    correct_predictions = 0

    print(f"\n\nstudents in {course_subject} {course_number}")

    for i in range(0, len(target_vector)):
        if target_vector[i] == 1:
            print(fall_2021_students_df.iloc[i].to_frame().T.to_string()) # student in course
        correct_predictions += target_vector[i] and y_pred[i] # correct predictions

    print(f"\nstudents predicted to be in {course_subject} {course_number}")

    for i in range(0, len(target_vector)):
        if y_pred[i] == 1:
            print(fall_2021_students_df.iloc[i].to_frame().T.to_string())

    print(f"{int(sum(target_vector))} students from 2021 took the course in 2022. {int(correct_predictions)} predictions were correct. there were {fp} false positives and {fn} false negatives.")

def main():
    # student_data = pd.read_excel("July 10 Dataset.xlsx")
    # pd.to_pickle(student_data, "July_10.pkl")

    student_data = pd.read_pickle("July_10.pkl")

    create_rf_model_for_course(student_data, "BSBA", "2209")

if __name__ == "__main__":
    main()